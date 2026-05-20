package com.ann.project;

import java.io.Serializable;
import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.stream.Stream;

public class SoftMax implements Activation, Serializable{
    @Override
    public double[][] apply(double[][] Z) {
    double [][] result = new double[Z.length][Z[0].length];

        for (int i = 0; i < Z.length; i++) {
            double max;
            try {
                max = Arrays.stream(Z[i]).max().getAsDouble();
            }catch (Exception e){
                max = 0;
                System.out.println("Error in softmax");
                for (int j = 0; j < Z[0].length; j++) {
                    Z[i][j] = 0;
                }
            }

            double sum = 0;
            for (int j = 0; j < Z[0].length; j++) {
                result[i][j] = Math.exp(Z[i][j]-max);
                sum += result[i][j];
            }
            for (int j = 0; j < Z[0].length; j++) {
                result[i][j] /= sum;
            }
        }
        return result;
    }

    @Override
    public double[][] derivative(double[][] A) {
    double [][] result = new double[A.length][A[0].length];
    for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                result[i][j] = 1;
            }
        }
        return result;
    }
}
