package com.ann.project;

import java.util.Random;

/*
We can write about how and why our neural network differs from the 4 graph types as the neural network looks a lot like a graph
My initial thoughts are that we only need to know the previous nodes so it would be unnessesary to store references between all of them

 */
public class NeuralNetwork{

    private final Random random;
    private int numberOfLayers;
    private int[] numHiddenLayers;
    private Layer[] network;
    private double[][] inputValues;
    private String[] classLabels;
    // Indexed parallel to network[]; activations[0] is the input layer's activation and unused.
    private Activation[] activations;
    private int batchSize;
    private double learningRate;

    /*
    The neural network is set up, now we need to begin to make the training logic

     */



    public NeuralNetwork(int seed,double learningRate, Activation[] activations,int batchSize ,int[] hiddenLayers, double[][] inputValues, String[] classLabels) {
        random = new Random(seed);
        this.learningRate = learningRate;
        this.numHiddenLayers = hiddenLayers;
        this.inputValues = inputValues;
        this.numberOfLayers = hiddenLayers.length + 2;
        this.classLabels = classLabels;
        this.activations = activations;
        this.batchSize = batchSize;

        createNetwork();
        populateLayers();

    }
    private void createNetwork(){
        Layer[] layers = new Layer[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {

            if (i == 0){
                layers[i] = new Layer(inputValues[0].length);
            }else if (i == numberOfLayers - 1){
                layers[i] = new Layer(classLabels.length);
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

    private void feedForward(Layer prevLayer, Layer currLayer){
        double[][] Z = NeuralUtil.dotMatrix(prevLayer.getA(), currLayer.getWeights());
        Z = NeuralUtil.matrixAddBias(Z, currLayer.getBias());
        double[][] A = currLayer.applyActivation();
        currLayer.setA(A);
        currLayer.setZ(Z);
    }

    private void train(){
        //
        double[][] label = new double[0][];

        //
        for (int i = 1; i < network.length; i++){
            feedForward(network[i-1], network[i]);
        }

        double batchCost = NeuralUtil.crossEntropy(label,network[network.length-1].getA());

        double[][] loss = NeuralUtil.matrixSubtract(label,network[network.length-1].getA());

        for (int i = network.length - 1; i > 0; i--) {
            loss = backPropagation(network[i], network[i + 1].getA(), loss);
        }

        for (int i = network.length - 1; i > 0; i--) {
            gradientDescent(network[i]);
        }

        //print cost maybe

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
        layerLoss = NeuralUtil.dotMatrix(NeuralUtil.transpose(layer.getWeights()),delta);

        layer.setWeightGradients(dZ_dW);
        layer.setBiasGradients(dZ_dB);

        return layerLoss;
    }

    public void gradientDescent(Layer layer){
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
}
