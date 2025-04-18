<!--- File created to explain NNPA hardware operations and limitations.  -->
<!-- This file was created manually refer to https://github.com/onnx/onnx-mlir/issues/3125 for more information -->

# Supported Operations for Target *NNPA*.

This document highlights operations that are performed on NNPA hardware that are not explicilty supported by ONNX. 

* **Minimum NNPA Level(Inclusive)** indicates the lowest and highest NNPA level a model may have for onnx-mlir to support compiling a model with the operator.
    * A ^ indicates onnx-mlir is compatible with the latest level of the NNPA Architecture which is z17.
    * Refer to [SupportedONNXOps-NNPA.md](https://github.com/onnx/onnx-mlir/blob/main/docs/SupportedONNXOps-NNPA.md) for ONNX supported operations.

* **Improvements**
    * Transposed MatMul - Optimization of the pattern MatMul followed by transpose consolidated and executed on NNPA.
    * Maximum Dimension Index Size (MDIS) - /*e1*/ 2097152, /*e2*/ 1048576, /*e3*/ 32768, /*e4*/ 32768.
    * Stickification - Perfroms data conversions to NNPAs internal format, DLFLOAT16, on the NNPA.
    * MatMul Broadcast - Adds Bcast1 support to the MatMul operation.

| Op |Minimum NNPA Level(Inclusive) |Limitations |Notes |
| --- |--- |--- |--- | 
| **Invsqrt** |z17 - ^ | Input tensor must be less than or equal to 4 dimensions. | | 