package com.ann.project;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/*
We can write about how and why our neural network differs from the 4 graph types as the neural network looks a lot like a graph
My initial thoughts are that we only need to know the previous nodes so it would be unnessesary to store references between all of them
 */

/*
    Need:
    Nice to have:
        parallelism
 */
public class NeuralNetwork implements Serializable{

    private Random random;
    private int epochs;
    private int batchSize;
    private double learningRate;
    private Activation[] activations;
    private int numberOfLayers;
    private int[] numHiddenLayers;
    private Layer[] network;
    private double[][] trainingData;
    private double[][] trainingLabels;
    private double[][] validationData;
    private double[][] ValidationLabels;
    // Indexed parallel to network[]; activations[0] is the input layer's activation and unused.
    private double lastTrainingCost;
    private double bestValCost = Double.MAX_VALUE;
    private Layer[] bestNetwork;
    private String filename;
    private boolean save;

    /*
    The neural network is set up, now we need to begin to make the training logic

     */



    public NeuralNetwork(int seed, int epochs, int batchSize, double learningRate, Activation[] activations, int[] hiddenLayers, double[][][] data, String filename, boolean save) {
        random = new Random(seed);
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.numHiddenLayers = hiddenLayers;
        this.numberOfLayers = hiddenLayers.length + 2;
        this.trainingData = data[0];
        this.trainingLabels = data[1];
        this.validationData = data[2];
        this.ValidationLabels = data[3];
        this.activations = activations;
        this.filename = filename;
        this.save = save;


        createNetwork();
        populateLayers();


    }

    public NeuralNetwork(Layer[] network) {
        this.network = network;
    }

    private void createNetwork(){
        Layer[] layers = new Layer[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            if (i == 0){
                layers[i] = new Layer(trainingData[0].length);
            }else if (i == numberOfLayers - 1){
                layers[i] = new Layer(trainingLabels[0].length);
                layers[i].initialise(layers[i-1].getNeurons(), activations[i], batchSize);
            }else{
                layers[i] = new Layer(numHiddenLayers[i-1]);
                layers[i].initialise(layers[i-1].getNeurons(), activations[i], batchSize);
            }
        }
        network = layers;
    }

    /*
    Biases are initialised as 0 on purpose
     */
    private void populateLayers(){
        for (int i = 1; i < network.length; i++){
            Layer layer = network[i];
            int fanIn = network[i-1].getNeurons();
            int fanOut = (network[i].getNeurons());
            for(int j = 0; j < layer.getWeights().length; j++){
                for(int k = 0; k < layer.getWeights()[j].length; k++){
                    layer.getWeights()[j][k] = (layer.getActivation() instanceof ReLU)
                            ? NeuralUtil.heInitialise(fanIn, random)
                            : NeuralUtil.xavierInitialise(fanIn, fanOut, random);
                }
            }
        }
    }

    private void populateInputLayer(double[][] inputValues){
        double[][] input = inputValues.clone();
            network[0].setA(input);
    }

    private void feedForward(Layer prevLayer, Layer currLayer){
        double[][] Z = NeuralUtil.dotMatrix(prevLayer.getA(), currLayer.getWeights());
        Z = NeuralUtil.matrixAddBias(Z, currLayer.getBias());
        currLayer.setZ(Z);
        double[][] A = currLayer.applyActivation();
        currLayer.setA(A);
    }

    public void trainBatch(){
        int indexLength = trainingData.length;
        int[] indices = new int[indexLength];
        for (int i = 0; i < indexLength; i++) {
            indices[i] = i;
        }
        Collections.shuffle(Arrays.asList(indices), random);
        for (int i = 0; i < indexLength / batchSize; i++) {
            double[][] batch = new double[batchSize][trainingData[0].length];
            double[][] label = new double[batchSize][trainingLabels[0].length];
            for (int j = 0; j < batchSize; j++) {
                int index = indices[i*batchSize+j];
                batch[j] = trainingData[index].clone();
                label[j] = trainingLabels[index].clone();
            }
            double batchCost = train(batch,label);
            lastTrainingCost = batchCost;
        }
    }

    public void trainEpoch(EpochListener listener) throws IOException {
        long totalTime = System.nanoTime();
        for (int i = 1; i < epochs+1; i++) {
            System.out.println("training Epoch "+ i);
            long epochTime = System.nanoTime();
            trainBatch();
            double loss = runInference(validationData, ValidationLabels);
            int accuracy;
            accuracy = getAccuracy(ValidationLabels, network[network.length-1].getA());

            if (listener != null) {
                listener.onEpoch(i, lastTrainingCost, loss, accuracy);
            }

            System.out.println("Epoch " + i + " Training loss: " + lastTrainingCost);
            System.out.println("Epoch Validation accuracy: " + accuracy + "%");
            System.out.println("Epoch " + i + " Validation loss: " + loss);

            if (isBest(loss) && save){
                bestValCost = loss;
                bestNetwork = network.clone();
                System.out.println("Epoch " + i + " is the best." + " \nBest Loss: " + loss);
                if (bestNetwork != null) saveNetwork(bestNetwork, "best_" + filename);

            }
            System.out.println("Epoch "+ i + " time: " + (System.nanoTime() - epochTime)/1000000 +"ms");
        }
        System.out.println("Total time: " + (System.nanoTime() - totalTime)/1000000 +"ms");
        if (save && network != null) saveNetwork(network.clone(), "latest_" + filename);
    }

    public double train(double[][] batch, double[][] label){
        double batchCost;
        double[][] loss;

        populateInputLayer(batch);

        for (int i = 1; i < network.length; i++){
            feedForward(network[i-1], network[i]);
        }

        batchCost = NeuralUtil.crossEntropy(label,network[network.length-1].getA());

        loss = NeuralUtil.matrixSubtract(label,network[network.length-1].getA());

        for (int i = network.length - 1; i > 0; i--) {
            loss = backPropagation(network[i], network[i - 1].getA(), loss);
        }

        for (int i = network.length - 1; i > 0; i--) {
            gradientDescent(network[i]);
        }
        return batchCost;
    }

    private double[][] backPropagation(Layer layer, double[][] prevA, double[][] loss){

        double[][] delta;
        double[][] dA_dZ;
        double[][] dZ_dW;
        double[] dZ_dB;
        double[][] layerLoss;

        //get derivative
        dA_dZ = layer.derivative();

        //calculate delta with hadamard product
        delta = NeuralUtil.hadamardProduct(loss, dA_dZ);

        //calculate weight gradiant
        dZ_dW = NeuralUtil.dotMatrix(NeuralUtil.transpose(prevA),delta);

        //average the weight gradiant
        dZ_dW = NeuralUtil.averageWeights(dZ_dW, batchSize);

        //calculate bias gradiant
        dZ_dB = NeuralUtil.biasGradiant(delta);

        //calculate loss in the current layer
        layerLoss = NeuralUtil.dotMatrix(delta, NeuralUtil.transpose(layer.getWeights()));

        layer.setWeightGradients(dZ_dW);
        layer.setBiasGradients(dZ_dB);

        return layerLoss;
    }

    private void gradientDescent(Layer layer){
        double[][] newWeight;
        double[] newBias;
        double[][] scalarWeight;
        double[] scalarBias;

        scalarWeight = NeuralUtil.scalarMultiply(learningRate, layer.getWeightGradients());

        newWeight = NeuralUtil.matrixSubtract(layer.getWeights(), scalarWeight);

        scalarBias = NeuralUtil.scalarMulVec(learningRate, layer.getBiasGradients());

        newBias = NeuralUtil.vectorSubtract(layer.getBias(), scalarBias);

        layer.setWeights(newWeight);
        layer.setBias(newBias);
    }

    private double runInference(double[][] inferenceData, double[][] inferenceLabels){
        double loss;
        populateInputLayer(inferenceData);
        for (int i = 1; i < network.length; i++) {
            feedForward(network[i - 1], network[i]);
        }
        loss = NeuralUtil.crossEntropy(inferenceLabels, network[network.length - 1].getA());
        return loss;
    }

    public void runInferenceSingle(double[][] inferenceData, double[][] inferenceLabels){
        for (int i = 0; i < inferenceLabels.length; i++) {

                double loss;

                double[][] singleDate = {inferenceData[i].clone()};
                double[][] singleLabel = {inferenceLabels[i].clone()};
                loss = runInference(singleDate, singleLabel);

                double[][] A = network[network.length-1].getA().clone();
                int lMax = NeuralUtil.getMaxValueIndex(singleLabel[0]);
                int pMax = NeuralUtil.getMaxValueIndex(A[0]);
                double[] tMax = Arrays.stream(A[0]).sorted().toArray().clone();
                OptionalDouble tcon = (Arrays.stream(A[0]).max());
                double confidence = tcon.getAsDouble() * 100;
                int secondMax = 0;
                int thirdMax = 0;
                for (int j = 0; j < tMax.length; j++) {
                    if (A[0][j] == tMax[tMax.length-2]){
                            secondMax += j;
                    }
                    if (A[0][j] == tMax[tMax.length-3]){
                        thirdMax += j;
                    }
                }
                System.out.println("Testing loss: " + loss);
                System.out.println("Thought it was " + pMax +" with " + (int)confidence + "%"  + " confidence.");
                System.out.println("the second and third max: " + secondMax + " and " + thirdMax + " respectively.");
                System.out.println("Correct Label: " + lMax);

        }
    }

    private int getAccuracy(double[][] labels, double[][] predictions){
        int result = 0;
        for (int i = 0; i < labels.length; i++) {
            int lMax = NeuralUtil.getMaxValueIndex(labels[i]);
            int pMax = NeuralUtil.getMaxValueIndex(predictions[i]);
            if (lMax == pMax) result++;
        }
        result = (result * 100) / labels.length;
        return result;
    }

    public boolean isBest(double cost){
        return cost < bestValCost;
    }

    public void saveNetwork(Layer[] network, String file) throws IOException {
        Files.createDirectories(Paths.get("Models"));
        FileOutputStream fos = new FileOutputStream("Models/"+file,false);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.flush();
        for (Layer layer : network) {
            oos.writeObject(layer);
        }
        oos.writeObject(null);
        oos.close();
    }

    public static Layer[] loadNetwork(String file) throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(file);
        ObjectInputStream ois = new ObjectInputStream(fis);
        ArrayList<Layer> layers = new ArrayList<>();
        Layer layer;
        while ((layer = (Layer)ois.readObject()) != null) {
            layers.add(layer);
        }
        ois.close();
        Layer[] network = new Layer[layers.size()];
        for (int i = 0; i < network.length; i++) {
            network[i] = layers.get(i);
        }
        return network;
    }
}
