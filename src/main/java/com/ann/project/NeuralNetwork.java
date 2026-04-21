package com.ann.project;

import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;

/*
We can write about how and why our neural network differs from the 4 graph types as the neural network looks a lot like a graph
My initial thoughts are that we only need to know the previous nodes so it would be unnessesary to store references between all of them

 */
public class NeuralNetwork{

    private final Random random;

    private int[] hiddenLayers;
    private ArrayList<Neuron> inputNeurons = new ArrayList<>();
    private ArrayList<ArrayList<Neuron>> hiddenNeurons = new ArrayList<>();
    private ArrayList<Neuron> outputNeurons = new ArrayList<>();
    private ArrayList<ArrayList<Neuron>> network = new ArrayList<>();


    /*
    The input is the dataset with the arraylist being a list of every datapoint, the double array containing each element of the datapoint.
    */
    private ArrayList<Double[]> inputValues = new ArrayList<>();

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
    private ArrayList<Object> outputValues = new ArrayList<>();

    public NeuralNetwork(int seed, int[] hiddenLayers, ArrayList<Double[]> inputValues, ArrayList<Object> outputValues) {
        random = new Random(seed);
        this.hiddenLayers = hiddenLayers;
        this.inputValues = inputValues;

        createInputNeurons();
        createHiddenNeurons();
        createOutputNeurons();

        network.add(inputNeurons);
        network.addAll(hiddenNeurons);
        network.add(outputNeurons);

        setPrevNeuron();
        setWeights();
        setBias();

    }

    private void createHiddenNeurons(){
        for (int hiddenLayer : hiddenLayers) {
            ArrayList<Neuron> neurons = new ArrayList<>();
            for (int j = 0; j < hiddenLayer; j++) {
                Neuron neuron = new Neuron();
                neurons.add(neuron);
            }
            hiddenNeurons.add(neurons);
        }
    }

    private void createInputNeurons(){
        for(int i = 0; i <= inputValues.get(0).length - 1; i++){
            Neuron neuron = new Neuron();
            inputNeurons.add(neuron);
        }
    }

    private void createOutputNeurons(){
        for(int i = 0; i < outputValues.size(); i++){
            Neuron neuron = new Neuron();
            outputNeurons.add(neuron);
        }
    }

    private void setPrevNeuron(){
        for (int i = 1; i < network.size(); i++) {
            for (int j = 0; j < network.get(i).size(); j++) {
                network.get(i).get(j).setPrevNeurons(network.get(i - 1));
            }
        }
    }

    private void setWeights(){
        for(ArrayList<Neuron> neurons : network){
            for(Neuron neuron : neurons){
                ArrayList<Double> weights = new ArrayList<>();
                for(int i = 0; i < neuron.getPrevNeurons().size(); i++){
                    weights.add(random.nextDouble());
                }
                neuron.setWeight(weights);
            }
        }
    }

    private void setBias(){
        for(ArrayList<Neuron> neurons : network){
            for(Neuron neuron : neurons){
                neuron.setBias(random.nextDouble());
            }
        }
    }

    private void feedForward(Neuron neuron){

    }
}
