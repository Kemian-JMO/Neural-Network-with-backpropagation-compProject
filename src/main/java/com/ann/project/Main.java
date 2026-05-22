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
        int[] hiddenLayers = {250,250,250};
        String best= "Models/best_";
        String name = "L3_N250-250-250_E10_AReLU_Lr0.01.nn";

        NeuralNetwork neuralNetwork = new NeuralNetwork(10,10,30,0.01, activations, hiddenLayers, data, name, true);
        neuralNetwork.trainEpoch((EpochListener) (epoch, trainLoss, valLoss, accuracy) -> {});

        Imager img = new Imager();
        double[][] testImages = img.getImageData();
        double[][] testLabels = img.getImageLabel();
        System.out.println("Network: " + name);
        NeuralNetwork bestNetwork = new NeuralNetwork(NeuralNetwork.loadNetwork(best + name));
        bestNetwork.runInferenceSingle(testImages, testLabels);

        System.out.println("Billeder indlæst: " + images.length);
        System.out.println("Labels indlæst: "   + labels.length);
        System.out.println("Første label: "     + labels[0]);
    }
}