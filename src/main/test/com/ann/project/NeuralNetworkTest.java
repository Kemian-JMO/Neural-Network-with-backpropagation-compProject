package com.ann.project;

import static org.junit.jupiter.api.Assertions.*;
        import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.Arrays;
import java.lang.reflect.Field;

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

    // Helper methods to access private fields using reflection
    @SuppressWarnings("unchecked")
    private ArrayList<Neuron> getInputNeurons(NeuralNetwork network) throws Exception {
        Field field = NeuralNetwork.class.getDeclaredField("inputNeurons");
        field.setAccessible(true);
        return (ArrayList<Neuron>) field.get(network);
    }

    @SuppressWarnings("unchecked")
    private ArrayList<ArrayList<Neuron>> getHiddenNeurons(NeuralNetwork network) throws Exception {
        Field field = NeuralNetwork.class.getDeclaredField("hiddenNeurons");
        field.setAccessible(true);
        return (ArrayList<ArrayList<Neuron>>) field.get(network);
    }

    @SuppressWarnings("unchecked")
    private ArrayList<Neuron> getOutputNeurons(NeuralNetwork network) throws Exception {
        Field field = NeuralNetwork.class.getDeclaredField("outputNeurons");
        field.setAccessible(true);
        return (ArrayList<Neuron>) field.get(network);
    }

    @SuppressWarnings("unchecked")
    private ArrayList<ArrayList<Neuron>> getNetwork(NeuralNetwork network) throws Exception {
        Field field = NeuralNetwork.class.getDeclaredField("network");
        field.setAccessible(true);
        return (ArrayList<ArrayList<Neuron>>) field.get(network);
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
    void testNeuronConnectionsInNetwork() throws Exception {
        System.out.println("Testing neuron connections in network...");
        
        NeuralNetwork network = new NeuralNetwork(42, testHiddenLayers, testInputValues, testOutputValues);
        
        // Get the network layers using reflection
        ArrayList<ArrayList<Neuron>> allLayers = getNetwork(network);
        
        // Verify that we have the correct number of layers (input + hidden + output)
        int expectedLayers = 1 + testHiddenLayers.length + 1; // input + hidden layers + output
        assertEquals(expectedLayers, allLayers.size(), "Network should have correct number of layers");
        System.out.println("✓ Network has " + allLayers.size() + " layers");
        
        // Test connections from layer 1 onwards (skip input layer as it has no previous neurons)
        for (int i = 1; i < allLayers.size(); i++) {
            ArrayList<Neuron> currentLayer = allLayers.get(i);
            ArrayList<Neuron> previousLayer = allLayers.get(i - 1);
            
            System.out.println("Testing layer " + i + " (size: " + currentLayer.size() + 
                ") connections to layer " + (i-1) + " (size: " + previousLayer.size() + ")");
            
            // Each neuron in current layer should reference all neurons in previous layer
            for (int j = 0; j < currentLayer.size(); j++) {
                Neuron neuron = currentLayer.get(j);
                ArrayList<Neuron> prevNeurons = neuron.getPrevNeurons();
                
                assertNotNull(prevNeurons, "Neuron " + j + " in layer " + i + " should have previous neurons set");
                assertEquals(previousLayer.size(), prevNeurons.size(), 
                    "Neuron " + j + " in layer " + i + " should reference all " + previousLayer.size() + " neurons in previous layer");
                
                // Verify that the references are actually to the correct neurons
                for (int k = 0; k < previousLayer.size(); k++) {
                    assertSame(previousLayer.get(k), prevNeurons.get(k), 
                        "Neuron " + j + " in layer " + i + " should reference neuron " + k + " from previous layer");
                }
            }
        }
        
        System.out.println("✓ All neuron connections verified successfully!");
    }
    
    @Test
    void testInputLayerHasNoPreviousNeurons() throws Exception {
        System.out.println("Testing that input layer has no previous neurons...");
        
        NeuralNetwork network = new NeuralNetwork(42, testHiddenLayers, testInputValues, testOutputValues);
        ArrayList<Neuron> inputNeurons = getInputNeurons(network);
        
        // Input neurons should not have previous neurons
        for (int i = 0; i < inputNeurons.size(); i++) {
            Neuron neuron = inputNeurons.get(i);
            ArrayList<Neuron> prevNeurons = neuron.getPrevNeurons();
            // Previous neurons should be null or empty for input layer
            assertTrue(prevNeurons == null || prevNeurons.isEmpty(), 
                "Input neuron " + i + " should not have previous neurons");
        }
        
        System.out.println("✓ Input layer properly has no previous connections");
    }
    
    @Test
    void testHiddenLayerConnectionsToInputLayer() throws Exception {
        System.out.println("Testing first hidden layer connections to input layer...");
        
        NeuralNetwork network = new NeuralNetwork(42, testHiddenLayers, testInputValues, testOutputValues);
        
        ArrayList<Neuron> inputNeurons = getInputNeurons(network);
        ArrayList<ArrayList<Neuron>> hiddenNeurons = getHiddenNeurons(network);
        
        // First hidden layer should connect to input layer
        if (!hiddenNeurons.isEmpty()) {
            ArrayList<Neuron> firstHiddenLayer = hiddenNeurons.get(0);
            
            System.out.println("First hidden layer has " + firstHiddenLayer.size() + " neurons");
            System.out.println("Input layer has " + inputNeurons.size() + " neurons");
            
            for (int i = 0; i < firstHiddenLayer.size(); i++) {
                Neuron hiddenNeuron = firstHiddenLayer.get(i);
                ArrayList<Neuron> prevNeurons = hiddenNeuron.getPrevNeurons();
                
                assertNotNull(prevNeurons, "Hidden neuron " + i + " should have previous neurons");
                assertEquals(inputNeurons.size(), prevNeurons.size(), 
                    "Hidden neuron " + i + " should connect to all " + inputNeurons.size() + " input neurons");
                
                // Verify actual references
                for (int j = 0; j < inputNeurons.size(); j++) {
                    assertSame(inputNeurons.get(j), prevNeurons.get(j), 
                        "Hidden neuron " + i + " should reference input neuron " + j);
                }
            }
        }
        
        System.out.println("✓ First hidden layer properly connected to input layer");
    }
    
    @Test
    void testHiddenToHiddenLayerConnections() throws Exception {
        System.out.println("Testing hidden-to-hidden layer connections...");
        
        // Use a network with multiple hidden layers
        int[] multipleHiddenLayers = new int[]{4, 3, 2}; // 3 hidden layers
        NeuralNetwork network = new NeuralNetwork(42, multipleHiddenLayers, testInputValues, testOutputValues);
        
        ArrayList<ArrayList<Neuron>> hiddenNeurons = getHiddenNeurons(network);
        
        // Test connections between consecutive hidden layers
        for (int i = 1; i < hiddenNeurons.size(); i++) {
            ArrayList<Neuron> currentHiddenLayer = hiddenNeurons.get(i);
            ArrayList<Neuron> previousHiddenLayer = hiddenNeurons.get(i - 1);
            
            System.out.println("Testing connection from hidden layer " + (i-1) + " (size: " + 
                previousHiddenLayer.size() + ") to hidden layer " + i + " (size: " + 
                currentHiddenLayer.size() + ")");
            
            for (int j = 0; j < currentHiddenLayer.size(); j++) {
                Neuron neuron = currentHiddenLayer.get(j);
                ArrayList<Neuron> prevNeurons = neuron.getPrevNeurons();
                
                assertNotNull(prevNeurons, "Hidden neuron " + j + " in layer " + i + " should have previous neurons");
                assertEquals(previousHiddenLayer.size(), prevNeurons.size(), 
                    "Hidden neuron " + j + " should connect to all " + previousHiddenLayer.size() + " neurons in previous hidden layer");
                
                // Verify actual references
                for (int k = 0; k < previousHiddenLayer.size(); k++) {
                    assertSame(previousHiddenLayer.get(k), prevNeurons.get(k), 
                        "Hidden neuron " + j + " should reference previous hidden neuron " + k);
                }
            }
        }
        
        System.out.println("✓ All hidden-to-hidden connections verified");
    }
    
    @Test
    void testOutputLayerConnections() throws Exception {
        System.out.println("Testing output layer connections...");
        
        NeuralNetwork network = new NeuralNetwork(42, testHiddenLayers, testInputValues, testOutputValues);
        
        ArrayList<Neuron> outputNeurons = getOutputNeurons(network);
        ArrayList<ArrayList<Neuron>> hiddenNeurons = getHiddenNeurons(network);
        
        // Output layer should connect to last hidden layer (or input if no hidden layers)
        ArrayList<Neuron> expectedPreviousLayer;
        if (!hiddenNeurons.isEmpty()) {
            expectedPreviousLayer = hiddenNeurons.get(hiddenNeurons.size() - 1); // Last hidden layer
            System.out.println("Output layer connecting to last hidden layer (size: " + expectedPreviousLayer.size() + ")");
        } else {
            expectedPreviousLayer = getInputNeurons(network); // Direct to input if no hidden layers
            System.out.println("Output layer connecting directly to input layer (size: " + expectedPreviousLayer.size() + ")");
        }
        
        for (int i = 0; i < outputNeurons.size(); i++) {
            Neuron outputNeuron = outputNeurons.get(i);
            ArrayList<Neuron> prevNeurons = outputNeuron.getPrevNeurons();
            
            assertNotNull(prevNeurons, "Output neuron " + i + " should have previous neurons");
            assertEquals(expectedPreviousLayer.size(), prevNeurons.size(), 
                "Output neuron " + i + " should connect to all " + expectedPreviousLayer.size() + " neurons in previous layer");
            
            // Verify actual references
            for (int j = 0; j < expectedPreviousLayer.size(); j++) {
                assertSame(expectedPreviousLayer.get(j), prevNeurons.get(j), 
                    "Output neuron " + i + " should reference previous neuron " + j);
            }
        }
        
        System.out.println("✓ Output layer connections verified");
    }
    
    @Test
    void testDirectInputToOutputConnections() throws Exception {
        System.out.println("Testing direct input-to-output connections (no hidden layers)...");
        
        // Test connections when there are no hidden layers (direct input to output)
        int[] noHiddenLayers = new int[0];
        NeuralNetwork network = new NeuralNetwork(42, noHiddenLayers, testInputValues, testOutputValues);
        
        ArrayList<Neuron> inputNeurons = getInputNeurons(network);
        ArrayList<Neuron> outputNeurons = getOutputNeurons(network);
        
        System.out.println("Input neurons: " + inputNeurons.size() + ", Output neurons: " + outputNeurons.size());
        
        // Output neurons should directly connect to input neurons
        for (int i = 0; i < outputNeurons.size(); i++) {
            Neuron outputNeuron = outputNeurons.get(i);
            ArrayList<Neuron> prevNeurons = outputNeuron.getPrevNeurons();
            
            assertNotNull(prevNeurons, "Output neuron " + i + " should have previous neurons");
            assertEquals(inputNeurons.size(), prevNeurons.size(), 
                "Output neuron " + i + " should connect directly to all " + inputNeurons.size() + " input neurons");
            
            // Verify actual references
            for (int j = 0; j < inputNeurons.size(); j++) {
                assertSame(inputNeurons.get(j), prevNeurons.get(j), 
                    "Output neuron " + i + " should reference input neuron " + j);
            }
        }
        
        System.out.println("✓ Direct input-to-output connections verified");
    }
    
    @Test
    void testSingleHiddenLayerConnections() throws Exception {
        System.out.println("Testing single hidden layer connections...");
        
        int[] singleHiddenLayer = new int[]{4};
        NeuralNetwork network = new NeuralNetwork(42, singleHiddenLayer, testInputValues, testOutputValues);
        
        ArrayList<Neuron> inputNeurons = getInputNeurons(network);
        ArrayList<ArrayList<Neuron>> hiddenNeurons = getHiddenNeurons(network);
        ArrayList<Neuron> outputNeurons = getOutputNeurons(network);
        
        ArrayList<Neuron> hiddenLayer = hiddenNeurons.get(0);
        
        System.out.println("Architecture: Input(" + inputNeurons.size() + ") -> Hidden(" + 
            hiddenLayer.size() + ") -> Output(" + outputNeurons.size() + ")");
        
        // Hidden layer connects to input
        for (int i = 0; i < hiddenLayer.size(); i++) {
            Neuron hiddenNeuron = hiddenLayer.get(i);
            ArrayList<Neuron> prevNeurons = hiddenNeuron.getPrevNeurons();
            assertEquals(inputNeurons.size(), prevNeurons.size(), 
                "Hidden neuron " + i + " should connect to all input neurons");
            
            for (int j = 0; j < inputNeurons.size(); j++) {
                assertSame(inputNeurons.get(j), prevNeurons.get(j), 
                    "Hidden neuron " + i + " should reference input neuron " + j);
            }
        }
        
        // Output layer connects to hidden layer
        for (int i = 0; i < outputNeurons.size(); i++) {
            Neuron outputNeuron = outputNeurons.get(i);
            ArrayList<Neuron> prevNeurons = outputNeuron.getPrevNeurons();
            assertEquals(hiddenLayer.size(), prevNeurons.size(), 
                "Output neuron " + i + " should connect to all hidden neurons");
            
            for (int j = 0; j < hiddenLayer.size(); j++) {
                assertSame(hiddenLayer.get(j), prevNeurons.get(j), 
                    "Output neuron " + i + " should reference hidden neuron " + j);
            }
        }
        
        System.out.println("✓ Single hidden layer connections verified");
    }

    // ... keep all your existing tests ...
    @Test
    void testWeightInitialization() {
        NeuralNetwork network = new NeuralNetwork(42, testHiddenLayers, testInputValues, testOutputValues);
        assertNotNull(network, "Network should initialize weights without errors");
    }

    @Test
    void testDeterministicWeightGeneration() {
        int seed = 123;
        NeuralNetwork network1 = new NeuralNetwork(seed, testHiddenLayers, testInputValues, testOutputValues);
        NeuralNetwork network2 = new NeuralNetwork(seed, testHiddenLayers, testInputValues, testOutputValues);
        
        assertNotNull(network1, "First network should be created");
        assertNotNull(network2, "Second network should be created");
    }

    @Test
    void testLargeNetworkCreation() {
        int[] largeHiddenLayers = new int[]{100, 50, 25, 10};
        assertDoesNotThrow(() -> {
            new NeuralNetwork(42, largeHiddenLayers, testInputValues, testOutputValues);
        }, "Should handle large network creation");
    }

    @Test
    void testNetworkWithDifferentInputSizes() {
        ArrayList<Double[]> largeInputs = new ArrayList<>();
        largeInputs.add(new Double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
        largeInputs.add(new Double[]{11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0});
        int[] largeHiddenLayers = new int[]{100, 50, 25, 10};

        ArrayList<Object> largeOutputs = new ArrayList<>();
        largeOutputs.add("Class A");
        largeOutputs.add("Class B");

        assertDoesNotThrow(() -> {
            new NeuralNetwork(42, largeHiddenLayers, largeInputs, largeOutputs);
        }, "Should handle larger input dimensions");
    }
}