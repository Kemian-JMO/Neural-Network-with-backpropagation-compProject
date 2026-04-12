package com.ann.project;

import java.util.Random;

public class WeightAndBias {
    private double weight;
    private double bias;
    private final Random seed;
    final Neuron leftNeuron;


    public WeightAndBias(Random seed, Neuron leftNeuron ){

        this.seed = seed;
        this.leftNeuron = leftNeuron;
        weight = this.seed.nextDouble();
        bias = this.seed.nextDouble();
    }



}
