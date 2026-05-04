package com.ann.project;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for NeuralUtil.dotMatrix.
 *
 * Convention under test: standard matrix multiplication, result = a * b.
 * a has shape [m][k], b has shape [k][n], result has shape [m][n].
 *
 * For the project's batch-first activation layout, this is invoked as
 * dotMatrix(prevA, weights) where:
 *   prevA  is [batchSize][prevNeurons]
 *   weights is [prevNeurons][neurons]
 *   result is [batchSize][neurons]
 */
class NeuralUtilTest {

    private static final double TOLERANCE = 1e-9;

    /**
     * Compares two 2D double arrays element-wise within TOLERANCE.
     * Fails fast at the first mismatch with a message identifying the index.
     */
    private static void assertMatrixEquals(double[][] expected, double[][] actual) {
        assertEquals(expected.length, actual.length, "row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i].length, actual[i].length,
                    "column count mismatch at row " + i);
            for (int j = 0; j < expected[i].length; j++) {
                assertEquals(expected[i][j], actual[i][j], TOLERANCE,
                        "value mismatch at [" + i + "][" + j + "]");
            }
        }
    }

    /**
     * The hand-worked example used throughout the design discussion:
     * one sample of 2 features feeding into a 3-neuron layer.
     * prevA [1][2] * weights [2][3] = Z [1][3].
     */
    @Test
    void dotMatrix_handWorkedForwardPassExample() {
        double[][] prevA = {
                {0.5, 0.8}
        };
        double[][] weights = {
                {0.1, 0.3, 0.5},
                {0.2, 0.4, 0.6}
        };
        double[][] expected = {
                {0.21, 0.47, 0.73}
        };

        double[][] result = NeuralUtil.dotMatrix(prevA, weights);

        assertMatrixEquals(expected, result);
    }

    /**
     * A * I = A. Catches accumulator-initialisation bugs and any case
     * where the function inadvertently transposes or rearranges A.
     */
    @Test
    void dotMatrix_multiplyByIdentityReturnsOriginal() {
        double[][] a = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        };
        double[][] identity = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0}
        };

        double[][] result = NeuralUtil.dotMatrix(a, identity);

        assertMatrixEquals(a, result);
    }

    /**
     * Non-square shapes [2][3] * [3][4] = [2][4].
     * Guards against bugs that only surface when m != k != n.
     */
    @Test
    void dotMatrix_nonSquareShapes() {
        double[][] a = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        };
        double[][] b = {
                {1.0, 2.0, 3.0, 4.0},
                {5.0, 6.0, 7.0, 8.0},
                {9.0, 10.0, 11.0, 12.0}
        };
        // computed by hand:
        // result[0][0] = 1*1 + 2*5 + 3*9 = 38
        // result[0][1] = 1*2 + 2*6 + 3*10 = 44
        // result[0][2] = 1*3 + 2*7 + 3*11 = 50
        // result[0][3] = 1*4 + 2*8 + 3*12 = 56
        // result[1][0] = 4*1 + 5*5 + 6*9 = 83
        // result[1][1] = 4*2 + 5*6 + 6*10 = 98
        // result[1][2] = 4*3 + 5*7 + 6*11 = 113
        // result[1][3] = 4*4 + 5*8 + 6*12 = 128
        double[][] expected = {
                {38.0, 44.0, 50.0, 56.0},
                {83.0, 98.0, 113.0, 128.0}
        };

        double[][] result = NeuralUtil.dotMatrix(a, b);

        assertMatrixEquals(expected, result);
    }

    /**
     * Multiple samples in a batch. Confirms that batch-first layout
     * works for batchSize > 1 -- each row of prevA is processed
     * independently against the same weight matrix.
     */
    @Test
    void dotMatrix_multipleSamplesInBatch() {
        double[][] prevA = {
                {0.5, 0.8},
                {1.0, 0.0},
                {0.0, 1.0}
        };
        double[][] weights = {
                {0.1, 0.3, 0.5},
                {0.2, 0.4, 0.6}
        };
        // row 0: same as the hand-worked example -> [0.21, 0.47, 0.73]
        // row 1: 1*weights[0] + 0*weights[1] = [0.1, 0.3, 0.5]
        // row 2: 0*weights[0] + 1*weights[1] = [0.2, 0.4, 0.6]
        double[][] expected = {
                {0.21, 0.47, 0.73},
                {0.1, 0.3, 0.5},
                {0.2, 0.4, 0.6}
        };

        double[][] result = NeuralUtil.dotMatrix(prevA, weights);

        assertMatrixEquals(expected, result);
    }

    /**
     * Mismatched inner dimensions should fail loudly rather than silently
     * producing garbage. The current implementation throws
     * ArrayIndexOutOfBoundsException, which is acceptable -- the contract
     * is that callers supply compatible shapes.
     */
    @Test
    void dotMatrix_mismatchedInnerDimensionsThrows() {
        double[][] a = {
                {1.0, 2.0}     // [1][2]
        };
        double[][] b = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0} // [3][3] -- inner dim 3 != a's inner dim 2
        };

        assertThrows(IllegalArgumentException.class,
                () -> NeuralUtil.dotMatrix(a, b));
    }
}