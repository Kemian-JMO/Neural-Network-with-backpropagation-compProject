package com.ann.project;

import static org.junit.jupiter.api.Assertions.*;
        import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.Arrays;

class NeuralNetworkTest {

    private ArrayList<Double[]> testInputValues;
    private ArrayList<Object> testOutputValues;
    private int[] testHiddenLayers;

    @BeforeEach
    void setUp() {
        // Setup test data
        testInputValues = new ArrayList<>();
        testInputValues.add(new Double[]{1.0, 2.0, 3.0});
        testInputValues.add(new Double[]{4.0, 5.0, 6.0});

        testOutputValues = new ArrayList<>();
        testOutputValues.add("Class A");
        testOutputValues.add("Class B");

        testHiddenLayers = new int[]{5, 3}; // 2 hidden layers with 5 and 3 neurons
    }

    @Test
    void testNeuralNetworkCreation() {
        // Test successful neural network creation
        NeuralNetwork network = new NeuralNetwork(42, testHiddenLayers, testInputValues, testOutputValues);
        assertNotNull(network, "Neural network should be created successfully");
    }

    @Test
    void testNeuralNetworkCreationWithEmptyHiddenLayers() {
        // Test creation with no hidden layers (direct input to output)
        int[] emptyHiddenLayers = new int[0];
        assertDoesNotThrow(() -> {
            new NeuralNetwork(42, emptyHiddenLayers, testInputValues, testOutputValues);
        }, "Should handle empty hidden layers without throwing exception");
    }

    @Test
    void testNeuralNetworkCreationWithSingleHiddenLayer() {
        // Test creation with single hidden layer
        int[] singleHiddenLayer = new int[]{4};
        assertDoesNotThrow(() -> {
            new NeuralNetwork(42, singleHiddenLayer, testInputValues, testOutputValues);
        }, "Should handle single hidden layer creation");
    }

    @Test
    void testWeightInitialization() {
        NeuralNetwork network = new NeuralNetwork(42, testHiddenLayers, testInputValues, testOutputValues);

        // Since weights are set using populateWeights method, let's test if neurons have weights
        // Note: This would require exposing the network structure or adding getter methods
        // For now, we'll test that the network doesn't throw exceptions during creation
        assertNotNull(network, "Network should initialize weights without errors");
    }

    @Test
    void testDeterministicWeightGeneration() {
        // Test that same seed produces same weights (deterministic behavior)
        int seed = 123;
        NeuralNetwork network1 = new NeuralNetwork(seed, testHiddenLayers, testInputValues, testOutputValues);
        NeuralNetwork network2 = new NeuralNetwork(seed, testHiddenLayers, testInputValues, testOutputValues);

        // Both networks should be created successfully with same seed
        assertNotNull(network1, "First network should be created");
        assertNotNull(network2, "Second network should be created");
        // Note: To fully test weight equality, we'd need getter methods for weights
    }

    @Test
    void testErrorHandlingNullInputValues() {
        // Test error handling for null input values
        assertThrows(Exception.class, () -> {
            new NeuralNetwork(42, testHiddenLayers, null, testOutputValues);
        }, "Should throw exception for null input values");
    }

    @Test
    void testErrorHandlingNullOutputValues() {
        // Test error handling for null output values
        assertThrows(Exception.class, () -> {
            new NeuralNetwork(42, testHiddenLayers, testInputValues, null);
        }, "Should throw exception for null output values");
    }

    @Test
    void testErrorHandlingNullHiddenLayers() {
        // Test error handling for null hidden layers
        assertThrows(Exception.class, () -> {
            new NeuralNetwork(42, null, testInputValues, testOutputValues);
        }, "Should throw exception for null hidden layers");
    }

    @Test
    void testErrorHandlingEmptyInputValues() {
        // Test error handling for empty input values
        ArrayList<Double[]> emptyInputs = new ArrayList<>();
        assertThrows(Exception.class, () -> {
            new NeuralNetwork(42, testHiddenLayers, emptyInputs, testOutputValues);
        }, "Should throw exception for empty input values");
    }

    @Test
    void testErrorHandlingEmptyOutputValues() {
        // Test error handling for empty output values
        ArrayList<Object> emptyOutputs = new ArrayList<>();
        assertThrows(Exception.class, () -> {
            new NeuralNetwork(42, testHiddenLayers, testInputValues, emptyOutputs);
        }, "Should throw exception for empty output values");
    }

    @Test
    void testErrorHandlingNegativeHiddenLayerSize() {
        // Test error handling for negative hidden layer sizes
        int[] negativeHiddenLayers = new int[]{5, -3, 2};
        assertThrows(Exception.class, () -> {
            new NeuralNetwork(42, negativeHiddenLayers, testInputValues, testOutputValues);
        }, "Should throw exception for negative hidden layer sizes");
    }

    @Test
    void testErrorHandlingZeroHiddenLayerSize() {
        // Test error handling for zero hidden layer sizes
        int[] zeroHiddenLayers = new int[]{5, 0, 2};
        assertThrows(Exception.class, () -> {
            new NeuralNetwork(42, zeroHiddenLayers, testInputValues, testOutputValues);
        }, "Should throw exception for zero hidden layer sizes");
    }

    @Test
    void testErrorHandlingMismatchedDataSizes() {
        // Test when input and output data have different sizes
        ArrayList<Double[]> mismatchedInputs = new ArrayList<>();
        mismatchedInputs.add(new Double[]{1.0, 2.0});

        ArrayList<Object> mismatchedOutputs = new ArrayList<>();
        mismatchedOutputs.add("Class A");
        mismatchedOutputs.add("Class B");
        mismatchedOutputs.add("Class C"); // One more output than inputs

        assertThrows(Exception.class, () -> {
            new NeuralNetwork(42, testHiddenLayers, mismatchedInputs, mismatchedOutputs);
        }, "Should throw exception for mismatched input/output data sizes");
    }

    @Test
    void testLargeNetworkCreation() {
        // Test creation of a larger network
        int[] largeHiddenLayers = new int[]{100, 50, 25, 10};
        assertDoesNotThrow(() -> {
            new NeuralNetwork(42, largeHiddenLayers, testInputValues, testOutputValues);
        }, "Should handle large network creation");
    }

    @Test
    void testNetworkWithDifferentInputSizes() {
        // Test with larger input dimensions
        ArrayList<Double[]> largeInputs = new ArrayList<>();
        largeInputs.add(new Double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
        largeInputs.add(new Double[]{11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0});

        ArrayList<Object> largeOutputs = new ArrayList<>();
        largeOutputs.add("Class A");
        largeOutputs.add("Class B");

        assertDoesNotThrow(() -> {
            new NeuralNetwork(42, testHiddenLayers, largeInputs, largeOutputs);
        }, "Should handle larger input dimensions");
    }
}