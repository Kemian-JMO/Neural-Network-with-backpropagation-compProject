package com.ann.project;

import java.util.Random;

public final class NeuralUtil {

    private NeuralUtil(){}

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

    public static double[][] matrixAddBias(double[][] matrix, double bias[]){
        double[][] result = new double[matrix.length][matrix[0].length + 1];
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[0].length; j++){
                result[i][j] = matrix[i][j] + bias[i];
            }
        }
        return result;
    }

    public static double xavierInitialise(int fanIn, int fanOut, Random r){
        double limit = Math.sqrt((double) 6 / (fanIn + fanOut));
        return (r.nextDouble() * 2 - 1) * limit;
    }

    public static double heInitialise(int fanIn, Random r){
        double limit = Math.sqrt((double) 2 / fanIn);
        return (r.nextGaussian() * 2 - 1) * limit;
    }

    public static double crossEntropy(double[][] yTrue, double[][] yHat){
        double loss = 0;
        double epsilon = 1e-11;
        for (int i = 0; i < yTrue.length; i++)
            for (int j = 0; j < yTrue[0].length; j++) {
                double yh = Double.isFinite(yHat[i][j]) ? yHat[i][j] : 0;
                yh = Math.max(yh, epsilon);
                yh = Math.log(yh);
                loss -= yTrue[i][j] * Math.log(yh);
            }
        return loss;
    }

    public static double[][] transpose(double[][] matrix){
        double[][] result = new double[matrix[0].length][matrix.length];
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[0].length; j++){
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    public static double[][] hadamardProduct(double[][] loss, double[][] derivative){
        double[][] hp = new double[loss.length][loss[0].length];
        for (int i = 0; i < loss.length; i++) {
            for(int j = 0; j < loss[0].length; j++){
              hp[i][j] = loss[i][j] * derivative[i][j];
            }
        }
        return hp;
    }

    public static double[][] averageWeights(double[][] gradiant, int batchSize){
        double[][] sum = new double[gradiant.length][gradiant[0].length];
        for(int i = 0; i < gradiant.length; i++){
            for(int j = 0; j < gradiant[0].length; j++){
                sum[i][j] = gradiant[i][j] / batchSize;
            }
        }
        return sum;
    }

    public static double[][] lossGradient(double[][] yTruth,double[][] yHat){
        double[][] loss = new double[yTruth.length][yTruth[0].length];
        for (int i = 0; i < yTruth.length; i++) {
            for(int j = 0; j < yTruth[0].length; j++){
                loss[i][j] = yHat[i][j] - yTruth[i][j];
            }
        }
        return loss;
    }

    public static double[] biasGradiant(double[][] delta){
        double[] gradiant = new double[delta[0].length];
        for(int i = 0; i < delta[0].length; i++){
            for(int j = 0; j < delta.length; j++){
                gradiant[i] += delta[j][i];
            }
            gradiant[i] /= delta.length;
        }
        return gradiant;
    }
}


