package com.ann.project;

import java.io.Serializable;

public class Sigmoid implements Activation, Serializable {

    @Override
    public double[][] apply(double[][] Z) {
        double[][] sigmoid = new double[Z.length][Z[0].length];
        for(int i = 0; i < sigmoid.length; i++){
            for(int j = 0; j < sigmoid[0].length; j++){
                sigmoid[i][j] = 1 / (1 + Math.exp(-Z[i][j]));
            }
        }
        return sigmoid;
    }

    @Override
    public double[][] derivative(double[][] A) {
        double[][] sigmoid = new double[A.length][A[0].length];
        for(int i = 0; i < sigmoid.length; i++){
            for(int j = 0; j < sigmoid[0].length; j++){
                sigmoid[i][j] = A[i][j] * (1 - A[i][j]);
            }
        }
        return sigmoid;
    }
}
