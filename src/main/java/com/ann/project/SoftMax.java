package com.ann.project;

public class SoftMax implements Activation{
    @Override
    public double[][] apply(double[][] Z) {
    double [][] result = new double[Z.length][Z[0].length];

        for (int i = 0; i < Z.length; i++) {
            double max = Z[i][0];
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
