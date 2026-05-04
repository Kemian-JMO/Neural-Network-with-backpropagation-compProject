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
    private Layer inputNeurons;
    private Layer[] hiddenLayers;
    private Layer outputLayer;
    private Layer[] network;


    /*
    The input is the dataset with the arraylist being a list of every datapoint, the double array containing each element of the datapoint.
    */
    private double[][] inputValues;

    /*
    the output values are objects, as we don't know what the output will be at this point.
    this could maybe be Strings?
     */

    /*
    The neural network is set up, now we need to begin to make the training logic
    feedforward:
    - set the input values as the activation values for the input neurons
    - calculate the activation values for the hidden neurons one layer at a time
    - this is done by multiplying the weights of the next layer by the activation values of the previous layer
    - by making a matrix multiplication of the weights and the activation values of the previous layer we can calculate the activation values of the next layer at once
    - then the bias is added by adding the bias to the product of the weights and the activation values of the previous layer in a matrix addition
    - the activation function is applied to the result of the matrix addition
        * should we have multiple activation functions and choose one?
        * it wouldnt be too hard to implement and we could have more to discuss about our results in the report
        * I have mostly read about Relu and sigmoid, for what i could read Relu should be better for object identification
    - calculate the activation values for the output neurons
    - compare the output values to the desired output values
    backpropagation:
    - calculate the error for each output neuron
    - calculate the error for each hidden neuron
    - update the weights and biases for each neuron

     */

    // Indexed parallel to network[]; activations[0] is the input layer's activation and unused.
    private String[] classLabels;
    private Activation[] activations;
    private int batchSize;

    public NeuralNetwork(int seed, Activation[] activations,int batchSize ,int[] hiddenLayers, double[][] inputValues, String[] classLabels) {
        random = new Random(seed);
        this.numHiddenLayers = hiddenLayers;
        this.inputValues = inputValues;
        this.numberOfLayers = hiddenLayers.length + 2;
        this.classLabels = classLabels;
        this.activations = activations;
        this.batchSize = batchSize;

        createNetwork();

    }

//    private void createHiddenLayers(){
//        Layer[] layers = new Layer[numHiddenLayers.length];
//        for (int i = 0; i < numHiddenLayers.length; i++) {
//            layers[i] = new Layer();
//            int prevSize = (i == 0) ? inputValues[0].length : numHiddenLayers[i - 1];
//            layers[i].initialise(numHiddenLayers[i], prevSize, activation);
//        }
//        hiddenLayers = layers;
//    }

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
            int fanOut = (i + 1 < network.length) ? network[i+1].getNeurons() : 0;
            for(int j = 0; j < layer.weights.length; j++){
                for(int k = 0; k < layer.weights[j].length; k++){
                    layer.weights[j][k] = (layer.activation instanceof ReLU)
                            ? NeuralUtil.heUniform(fanIn, random)
                            : NeuralUtil.xavierUniform(fanIn, fanOut, random);
                }
            }
        }
    }

    private void feedForward(Layer prevLayer, Layer currLayer){
        double[][] Z = NeuralUtil.dotMatrix(prevLayer.getA(), currLayer.getWeights());
    }
}
