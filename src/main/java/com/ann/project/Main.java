package com.ann.project;

public class Main {

    public static void main(String[] args) throws Exception {

        System.out.println("Indlæser MNIST data...");

        double[][] images = MnistLoader.loadImages(MnistLoader.TRAIN_IMAGE_FILE);
        double[][]      labels = MnistLoader.loadLabels(MnistLoader.TRAIN_LABEL_FILE);
        System.out.println(images[0].length);

        Activation RELU = new ReLU();
        Activation SOFTMAX = new SoftMax();
        Activation[] activations = {null, RELU, SOFTMAX};
        int[] hiddenLayers = {16};

        NeuralNetwork neuralNetwork = new NeuralNetwork(10,0.01, activations,10, hiddenLayers, images, labels);
        neuralNetwork.trainBatch();

        System.out.println("Billeder indlæst: " + images.length);
        System.out.println("Labels indlæst: "   + labels.length);
        System.out.println("Første label: "      + labels[0]);
    }

}