package com.ann.project;

import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;

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
    private ArrayList<Objects> outputValues = new ArrayList<>();


    public NeuralNetwork(int seed, int[] hiddenLayers, ArrayList<Double[]> inputValues, ArrayList<Objects> outputValues ){
        random = new Random(seed);
        this.hiddenLayers = hiddenLayers;
        this.inputValues = inputValues;

        createInputNeurons();
        createHiddenNeurons();
        createOutputNeurons();

        network.add(inputNeurons);
        network.addAll(hiddenNeurons);
        network.add(outputNeurons);

        /*
        This is the part where we set the prevNeurons of each neuron.
         */
//        for(int i = 1; i < network.size(); i++){
//            for(int j = 0; j < network.get(i).size(); j++){
//                network.get(i).get(j).setPrevNeurons(network.get(i-1));
//            }
//        }

        /*
        This is the part where we set the weights of each neuron.
         */

        /*
        This is the part where we set the bias of each neuron.
         */

        /*
        This is the part where we set the activation function of each neuron in the first layer.
         */
    }

    private void createHiddenNeurons(){
        for(int i = 0; i < hiddenLayers.length; i++){
            ArrayList<Neuron> neurons = new ArrayList<>();
            for (int j = 0; j < hiddenLayers[i]; j++) {
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

    private ArrayList<Double> populateWeights(int Size){
        ArrayList<Double> weights = new ArrayList<>();
        for(int i = 0; i < Size; i++){
            weights.add(random.nextDouble());
        }
        return weights;
    }
}
