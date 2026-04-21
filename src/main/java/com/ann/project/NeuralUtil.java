package com.ann.project;

public final class NeuralUtil {

    private NeuralUtil(){}

    public static double sigmoid(double x){
        return 1/(1+Math.exp(-x));
    }

    public static double[][] dotMatrix(double[][] a, double[][] b){
        double[][] result = new double[a.length][b[0].length];
        for(int i = 0; i < a.length; i++){
            for(int j = 0; j < b[0].length; j++){
                for(int k = 0; k < a[0].length; k++){
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }
}
