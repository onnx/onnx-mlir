// SPDX-License-Identifier: Apache-2.0

package com.ibm.onnxmlir;

import java.util.Arrays;

public class OMTensorTest {

    private static int testsRun = 0;
    private static int testsFailed = 0;

    private static void assertTrue(boolean condition, String message) {
        testsRun++;
        if (!condition) {
            System.err.println("FAIL: " + message);
            testsFailed++;
        }
    }

    private static void testValidCreation() {
        float[] data = new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        long[] shape = new long[] {2, 3};
        OMTensor omt = new OMTensor(data, shape);
        assertTrue(omt.getRank() == 2, "Rank should be 2");
        assertTrue(omt.getNumElems() == 6, "NumElems should be 6");
        assertTrue(omt.getDataType() == OMTensor.ONNX_TYPE_FLOAT, "Data type should be FLOAT");
        assertTrue(omt.getBufferSize() == 24, "Buffer size should be 24 bytes (6 * 4)");
        assertTrue(Arrays.equals(omt.getFloatData(), data), "Data content should match");
    }

    private static void testShapeDataMismatchThrows() {
        float[] data = new float[] {1.0f, 2.0f, 3.0f};
        long[] shape = new long[] {2, 3}; // requires 6 elements
        boolean threw = false;
        try {
            new OMTensor(data, shape);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        assertTrue(threw, "Mismatched shape and data should throw IllegalArgumentException");
    }

    private static void testNegativeDimensionThrows() {
        float[] data = new float[] {1.0f, 2.0f};
        long[] shape = new long[] {-1, 2};
        boolean threw = false;
        try {
            new OMTensor(data, shape);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        assertTrue(threw, "Negative dimension in shape should throw IllegalArgumentException");
    }

    private static void testOverflowingShapeThrows() {
        float[] data = new float[0];
        long[] shape = new long[] {Long.MAX_VALUE, 2};
        boolean threw = false;
        try {
            new OMTensor(data, shape);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        assertTrue(threw, "Overflowing shape product should throw IllegalArgumentException");
    }

    private static void testZeroDimensionAccepted() {
        float[] data = new float[0];
        long[] shape = new long[] {0, 5};
        OMTensor omt = new OMTensor(data, shape);
        assertTrue(omt.getRank() == 2, "Rank should be 2 for {0, 5}");
        assertTrue(omt.getNumElems() == 0, "NumElems should be 0 for {0, 5}");
        assertTrue(omt.getBufferSize() == 0, "Buffer size should be 0 for empty tensor");
    }

    private static void testShapeDefensiveClone() {
        float[] data = new float[] {1.0f, 2.0f};
        long[] shape = new long[] {1, 2};
        OMTensor omt = new OMTensor(data, shape);
        long[] shapeCopy = omt.getShape();
        shapeCopy[0] = 999;
        assertTrue(omt.getShape()[0] == 1, "Mutating returned getShape() array should not affect OMTensor");
    }

    public static void main(String[] args) {
        testValidCreation();
        testShapeDataMismatchThrows();
        testNegativeDimensionThrows();
        testOverflowingShapeThrows();
        testZeroDimensionAccepted();
        testShapeDefensiveClone();

        System.out.println("Ran " + testsRun + " tests, " + testsFailed + " failures.");
        if (testsFailed > 0) {
            System.exit(1);
        }
    }
}
