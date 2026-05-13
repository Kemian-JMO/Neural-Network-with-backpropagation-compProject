package com.ann.project;

public class Main {

    public static void main(String[] args) throws Exception {

        System.out.println("Indlæser MNIST data...");

        double[][] images = MnistLoader.loadImages(MnistLoader.TRAIN_IMAGE_FILE);
        double[][] labels = MnistLoader.loadLabels(MnistLoader.TRAIN_LABEL_FILE);
        double[][] testImages = MnistLoader.loadImages(MnistLoader.TEST_IMAGE_FILE);
        double[][] testLabels = MnistLoader.loadLabels(MnistLoader.TEST_LABEL_FILE);
        double[][][] data = {images, labels, testImages, testLabels};
        System.out.println(images[0].length);

        Activation RelU = new ReLU();
        Activation SOFTMAX = new SoftMax();
        Activation[] activations = {null, RelU, RelU, SOFTMAX};
        int[] hiddenLayers = {20,20};

        NeuralNetwork neuralNetwork = new NeuralNetwork(10,10,30,0.01, activations, hiddenLayers, data);
        neuralNetwork.trainEpoch();

        NeuralNetwork bestNetwork = new NeuralNetwork(NeuralNetwork.loadNetwork("Models/network.nn"));
        System.out.println("Billeder indlæst: " + images.length);
        System.out.println("Labels indlæst: "   + labels.length);
        System.out.println("Første label: "     + labels[0]);
    }
}