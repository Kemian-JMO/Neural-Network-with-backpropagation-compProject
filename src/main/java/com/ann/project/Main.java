package com.ann.project;

public class Main {

    public static void main(String[] args) throws Exception {

        System.out.println("Indlæser MNIST data...");

        double[][] images = MnistLoader.loadImages(MnistLoader.TRAIN_IMAGE_FILE);
        double[][] labels = MnistLoader.loadLabels(MnistLoader.TRAIN_LABEL_FILE);
        double[][] valImages = MnistLoader.loadImages(MnistLoader.TEST_IMAGE_FILE);
        double[][] valLabels = MnistLoader.loadLabels(MnistLoader.TEST_LABEL_FILE);
        double[][][] data = {images, labels, valImages, valLabels};
        System.out.println(images[0].length);

        
        Activation RelU = new ReLU();
        Activation Sigmoid = new Sigmoid();
        Activation SOFTMAX = new SoftMax();
        Activation[] activations = {null, RelU, RelU, RelU, SOFTMAX};
        int[] hiddenLayers = {200,200,200};

        NeuralNetwork neuralNetwork = new NeuralNetwork(10,10,30,0.01, activations, hiddenLayers, data);
        neuralNetwork.trainEpoch();


        Imager img = new Imager();
        double[][] testImages = img.getImageData();
        double[][] testLabels = img.getImageLabel();
        NeuralNetwork bestNetwork = new NeuralNetwork(NeuralNetwork.loadNetwork("Models/network.nn"));
        bestNetwork.runInferenceSingle(testImages, testLabels);
        System.out.println("Billeder indlæst: " + images.length);
        System.out.println("Labels indlæst: "   + labels.length);
        System.out.println("Første label: "     + labels[0]);
    }
}