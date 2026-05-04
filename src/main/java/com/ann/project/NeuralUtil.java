package com.ann.project;

import java.util.Random;

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
                    if (a[0].length != b.length) {
                        throw new IllegalArgumentException("Arrays have different lengths");
                    }
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    public static double xavierUniform(int fanIn, int fanOut, Random r){
        double limit = Math.sqrt((double) 6 / (fanIn + fanOut));
        return (r.nextDouble() * 2 - 1) * limit;
    }

    public static double heUniform(int fanIn, Random r){
        double limit = Math.sqrt((double) 6 / fanIn);
        return (r.nextDouble() * 2 - 1) * limit;
    }
}


