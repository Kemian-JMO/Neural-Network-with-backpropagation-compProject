package com.ann.project;

/*
does not work but was needed for the testing to work
 */
public class ReLU implements Activation{

    @Override
    public double[][] apply(double[][] Z) {
        return new double[0][];
    }

    @Override
    public double[][] derivative(double[][] Z, double[][] A) {
        return new double[0][];
    }
}
