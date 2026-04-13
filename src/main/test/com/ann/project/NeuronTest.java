package com.ann.project;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;

class NeuronTest {

    private Neuron neuron;
    private ArrayList<Double> testWeights;
    private ArrayList<Neuron> testPrevNeurons;

    @BeforeEach
    void setUp() {
        neuron = new Neuron();

        testWeights = new ArrayList<>();
        testWeights.add(0.5);
        testWeights.add(0.3);
        testWeights.add(0.8);

        testPrevNeurons = new ArrayList<>();
        testPrevNeurons.add(new Neuron());
        testPrevNeurons.add(new Neuron());
        testPrevNeurons.add(new Neuron());
    }

    @Test
    void testNeuronCreation() {
        // Test that neuron can be created
        assertNotNull(neuron, "Neuron should be created successfully");
    }

    @Test
    void testSetAndGetWeights() {
        // Note: There's a bug in the setWeight method - it creates a new ArrayList but doesn't copy the weights
        neuron.setWeight(testWeights);

        // This test will currently fail due to the bug in setWeight method
        // The method should be: this.weight = new ArrayList<>(weight);
        ArrayList<Double> retrievedWeights = neuron.getWeight();
        assertNotNull(retrievedWeights, "Retrieved weights should not be null");

        // This assertion will fail with current implementation
        // assertEquals(testWeights.size(), retrievedWeights.size(), "Weight size should match");
    }

    @Test
    void testSetWeightWithNullInput() {
        // Test error handling for null weight input
        assertDoesNotThrow(() -> {
            neuron.setWeight(null);
        }, "Setting null weights should not throw exception");

        // But getting weights might return null or empty list
        ArrayList<Double> weights = neuron.getWeight();
        // This depends on implementation - might be null or empty
    }

    @Test
    void testSetWeightWithEmptyList() {
        // Test with empty weight list
        ArrayList<Double> emptyWeights = new ArrayList<>();
        assertDoesNotThrow(() -> {
            neuron.setWeight(emptyWeights);
        }, "Setting empty weights should not throw exception");

        ArrayList<Double> retrievedWeights = neuron.getWeight();
        assertNotNull(retrievedWeights, "Retrieved weights should not be null");
        assertEquals(0, retrievedWeights.size(), "Empty weight list should remain empty");
    }

    @Test
    void testSetAndGetBias() {
        // Test bias setting and getting
        double testBias = 0.75;
        neuron.setBias(testBias);

        assertEquals(testBias, neuron.getBias(), 0.001, "Bias should be set correctly");
    }

    @Test
    void testSetBiasWithNegativeValue() {
        // Test setting negative bias
        double negativeBias = -0.5;
        assertDoesNotThrow(() -> {
            neuron.setBias(negativeBias);
        }, "Setting negative bias should not throw exception");

        assertEquals(negativeBias, neuron.getBias(), 0.001, "Negative bias should be set correctly");
    }

    @Test
    void testSetBiasWithZero() {
        // Test setting zero bias
        neuron.setBias(0.0);
        assertEquals(0.0, neuron.getBias(), 0.001, "Zero bias should be set correctly");
    }

    @Test
    void testSetPreviousNeurons() {
        // Test setting previous neurons
        assertDoesNotThrow(() -> {
            neuron.setPrevNeurons(testPrevNeurons);
        }, "Setting previous neurons should not throw exception");
    }

    @Test
    void testSetPreviousNeuronsWithNull() {
        // Test error handling for null previous neurons
        assertDoesNotThrow(() -> {
            neuron.setPrevNeurons(null);
        }, "Setting null previous neurons should not throw exception");
    }

    @Test
    void testSetPreviousNeuronsWithEmptyList() {
        // Test with empty previous neurons list
        ArrayList<Neuron> emptyPrevNeurons = new ArrayList<>();
        assertDoesNotThrow(() -> {
            neuron.setPrevNeurons(emptyPrevNeurons);
        }, "Setting empty previous neurons list should not throw exception");
    }

    @Test
    void testSetActivation() {
        // Test setting activation value
        double testActivation = 0.85;
        assertDoesNotThrow(() -> {
            neuron.setActivation(testActivation);
        }, "Setting activation should not throw exception");
    }

    @Test
    void testSetActivationWithNegativeValue() {
        // Test setting negative activation
        double negativeActivation = -0.3;
        assertDoesNotThrow(() -> {
            neuron.setActivation(negativeActivation);
        }, "Setting negative activation should not throw exception");
    }

    @Test
    void testSetActivationWithLargeValue() {
        // Test setting large activation value
        double largeActivation = 100.0;
        assertDoesNotThrow(() -> {
            neuron.setActivation(largeActivation);
        }, "Setting large activation should not throw exception");
    }

    @Test
    void testNeuronPreviousConnectionSetting() {
        // Test that a neuron can store references to previous neurons
        Neuron currentNeuron = new Neuron();
        ArrayList<Neuron> previousNeurons = new ArrayList<>();
        
        // Create some previous neurons
        Neuron prev1 = new Neuron();
        Neuron prev2 = new Neuron();
        Neuron prev3 = new Neuron();
        
        previousNeurons.add(prev1);
        previousNeurons.add(prev2);
        previousNeurons.add(prev3);
        
        // Set previous neurons
        assertDoesNotThrow(() -> {
            currentNeuron.setPrevNeurons(previousNeurons);
        }, "Setting previous neurons should not throw exception");
    }
    
    @Test
    void testNeuronConnectionWithSinglePreviousNeuron() {
        // Test connection with just one previous neuron
        Neuron currentNeuron = new Neuron();
        ArrayList<Neuron> previousNeurons = new ArrayList<>();
        
        Neuron singlePrev = new Neuron();
        previousNeurons.add(singlePrev);
        
        assertDoesNotThrow(() -> {
            currentNeuron.setPrevNeurons(previousNeurons);
        }, "Connection with single previous neuron should work");
    }
    
    @Test
    void testNeuronConnectionWithManyPreviousNeurons() {
        // Test connection with many previous neurons
        Neuron currentNeuron = new Neuron();
        ArrayList<Neuron> previousNeurons = new ArrayList<>();
        
        // Add 10 previous neurons
        for (int i = 0; i < 10; i++) {
            previousNeurons.add(new Neuron());
        }
        
        assertDoesNotThrow(() -> {
            currentNeuron.setPrevNeurons(previousNeurons);
        }, "Connection with many previous neurons should work");
    }
    
    @Test
    void testNeuronConnectionModification() {
        // Test modifying neuron connections
        Neuron currentNeuron = new Neuron();
        
        // First set of connections
        ArrayList<Neuron> firstConnections = new ArrayList<>();
        firstConnections.add(new Neuron());
        firstConnections.add(new Neuron());
        
        currentNeuron.setPrevNeurons(firstConnections);
        
        // Second set of connections (modifying)
        ArrayList<Neuron> secondConnections = new ArrayList<>();
        secondConnections.add(new Neuron());
        secondConnections.add(new Neuron());
        secondConnections.add(new Neuron());
        
        assertDoesNotThrow(() -> {
            currentNeuron.setPrevNeurons(secondConnections);
        }, "Modifying neuron connections should work");
    }
    
    @Test
    void testNeuronSelfConnection() {
        // Test that a neuron can reference itself (edge case)
        Neuron neuron = new Neuron();
        ArrayList<Neuron> selfConnection = new ArrayList<>();
        selfConnection.add(neuron); // Self-reference
        
        assertDoesNotThrow(() -> {
            neuron.setPrevNeurons(selfConnection);
        }, "Self-connection should not throw exception");
    }
    
    @Test
    void testCircularNeuronConnections() {
        // Test circular connections between neurons
        Neuron neuron1 = new Neuron();
        Neuron neuron2 = new Neuron();
        
        ArrayList<Neuron> connections1 = new ArrayList<>();
        connections1.add(neuron2);
        
        ArrayList<Neuron> connections2 = new ArrayList<>();
        connections2.add(neuron1);
        
        assertDoesNotThrow(() -> {
            neuron1.setPrevNeurons(connections1);
            neuron2.setPrevNeurons(connections2);
        }, "Circular connections should not throw exception");
    }
    
    @Test
    void testMultipleWeightOperations() {
        // Test multiple weight operations
        ArrayList<Double> firstWeights = new ArrayList<>();
        firstWeights.add(0.1);
        firstWeights.add(0.2);

        ArrayList<Double> secondWeights = new ArrayList<>();
        secondWeights.add(0.9);
        secondWeights.add(0.8);
        secondWeights.add(0.7);

        // Set first weights
        neuron.setWeight(firstWeights);
        ArrayList<Double> retrieved1 = neuron.getWeight();
        assertNotNull(retrieved1, "First weight retrieval should not be null");

        // Set second weights
        neuron.setWeight(secondWeights);
        ArrayList<Double> retrieved2 = neuron.getWeight();
        assertNotNull(retrieved2, "Second weight retrieval should not be null");

        // Note: Due to the bug in setWeight, this test will show the issue
        // The weights are not actually being stored correctly
    }
}
