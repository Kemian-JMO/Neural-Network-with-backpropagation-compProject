package com.ann.project;

import java.util.Random;

public class Weight {
    private double weight;
    private final Random seed;
    final Neuron leftNeuron;


    public Weight(Random seed, Neuron leftNeuron ){

        this.seed = seed;
        this.leftNeuron = leftNeuron;
        weight = this.seed.nextDouble();
    }



}
