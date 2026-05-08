package com.ann.project;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Random;

/*
Look into Java streams
 */
/*
This class is used to represent a neuron.
The weight
 */
public class Neuron {

    //private ArrayList<WeightAndBias> weightAndBias;
    private ArrayList<Double> weight;
    private double bias;
    private ArrayList<Neuron> prevNeurons;
    private double activation;

    public Neuron(){
    }

    public void setWeight(ArrayList<Double> weight){
        this.weight = weight;
    }

    public ArrayList<Double> getWeight(){
        return weight;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getBias() {
        return bias;
    }

    public void setPrevNeurons(ArrayList<Neuron> prevNeurons){
        this.prevNeurons = prevNeurons;
    }

    public ArrayList<Neuron> getPrevNeurons(){
        return prevNeurons;
    }

    public void setActivation(double activation){
        this.activation = activation;
    }

}
