package com.ann.project;

public class Layer {
    int neurons;
    double[][] weights;
    double[][] weightGradients;
    double[] bias;
    double[] biasGradients;
    double[][] A;
    double[][] Z;
    double[] deltas;
    Activation activation;

    Layer(int neurons){
        this.neurons = neurons;
    }

    public void initialise(int prevNeurons, Activation activation,int batchSize) {
        weights = new double[prevNeurons][neurons];
        weightGradients = new double[prevNeurons][neurons];
        bias = new double[neurons];
        biasGradients = new double[neurons];
        A = new double[batchSize][neurons];
        Z = new double[batchSize][neurons];
        deltas = new double[neurons];
        this.activation = activation;
    }

    public int getNeurons() {
        return neurons;
    }

    public void setNeurons(int neurons) {
        this.neurons = neurons;
    }

    public double[][] getWeights() {
        return weights;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public double[] getBias() {
        return bias;
    }

    public void setBias(double[] bias) {
        this.bias = bias;
    }

    public double[][] getA() {
        return A;
    }

    public void setA(double[][] a) {
        A = a;
    }

    public double[][] getZ() {
        return Z;
    }

    public void setZ(double[][] z) {
        Z = z;
    }

    public double[] getDeltas() {
        return deltas;
    }

    public void setDeltas(double[] deltas) {
        this.deltas = deltas;
    }

    public Activation getActivation() {
        return activation;
    }

    public void setActivation(Activation activation) {
        this.activation = activation;
    }

    public double[][] getWeightGradients() {
        return weightGradients;
    }

    public void setWeightGradients(double[][] weightGradients) {
        this.weightGradients = weightGradients;
    }

    public double[] getBiasGradients() {
        return biasGradients;
    }

    public void setBiasGradients(double[] biasGradients) {
        this.biasGradients = biasGradients;
    }

    public double[][] derivative(){
        return activation.derivative(A);
    }
}
