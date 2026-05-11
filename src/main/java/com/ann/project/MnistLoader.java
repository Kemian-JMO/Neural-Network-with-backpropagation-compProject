package com.ann.project;

import java.io.*;

public class MnistLoader {

    public static final int IMAGE_MAGIC_NUMBER = 2051;
    public static final int LABEL_MAGIC_NUMBER = 2049;

    public static final String TRAIN_IMAGE_FILE = "src/main/resources/com.mnist.training/train-images.idx3-ubyte";
    public static final String TRAIN_LABEL_FILE = "src/main/resources/com.mnist.training/train-labels.idx1-ubyte";
    public static final String TEST_IMAGE_FILE  = "src/main/resources/com.mnist.testing/t10k-images.idx3-ubyte";
    public static final String TEST_LABEL_FILE  = "src/main/resources/com.mnist.testing/t10k-labels.idx1-ubyte";

    public static double[][] loadImages(String filePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            int magicNumber = dis.readInt();
            if (magicNumber != IMAGE_MAGIC_NUMBER) {
                throw new IOException("Ugyldig MNIST billedfil.");
            }
            int numImages = dis.readInt();
            int numRows   = dis.readInt();
            int numCols   = dis.readInt();

            double[][] images = new double[numImages][numRows * numCols];

            for (int i = 0; i < numImages; i++) {
                for (int j = 0; j < numRows * numCols; j++) {
                    images[i][j] = dis.readUnsignedByte() / 255.0;
                }
            }
            return images;
        }
    }

    public static int[] loadLabels(String filePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            int magicNumber = dis.readInt();
            if (magicNumber != LABEL_MAGIC_NUMBER) {
                throw new IOException("Ugyldig MNIST labelfil.");
            }
            int numLabels = dis.readInt();

            int[] labels = new int[numLabels];

            for (int i = 0; i < numLabels; i++) {
                labels[i] = dis.readUnsignedByte();
            }
            return labels;
        }
    }

}