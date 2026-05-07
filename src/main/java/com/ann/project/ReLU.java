package com.ann.project;

/*
does not work but was needed for the testing to work
 */
public class ReLU implements Activation{

    @Override
    public double[][] apply(double[][] Z) {
        double[][] result = new double[Z.length][Z[0].length];
        for(int i = 0; i < result.length; i++){
            for(int j = 0; j < result[0].length; j++){
                double v = Double.isFinite(Z[i][j]) ? Z[i][j] : 0;
                result[i][j] = Math.max(0, v);
            }
        }
        return result;
    }

    @Override
    public double[][] derivative(double[][] A) {
        double[][] result = new double[A.length][A[0].length];
        for(int i = 0; i < result.length; i++){
            for(int j = 0; j < result[0].length; j++){
                double v = Double.isFinite(A[i][j]) ? A[i][j] : 0;
                A[i][j] = (v > 0) ? 1 : 0;
            }
        }
        return A;
    }
}
