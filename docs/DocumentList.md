<!--- SPDX-License-Identifier: Apache-2.0 -->

# Index of documents
This document serves as an index for onnx-mlir documents.

# Supported ONNX Ops
* CPU support is covered [here](SupportedONNXOps-cpu.md).
* NNPA support is covered [here](SupportedONNXOps-NNPA.md).

# Working environment
* Installation is covered by [README.md](../README.md).
* [Workflow.md](Workflow.md) describes how to contribute in github environment.
* [This guideline](Documentation.md) is used to keep documentation and code consistent.

# Development
* Onnx operation are represented with  [ONNX dialect](Dialects/onnx.md) in onnx-mlir.
*  This [document](ImportONNXDefs.md#add_operation)
tell you how to generate an ONNX operation into ONNX dialect.
* After an ONNX model is imported into onnx-mlir, several graph-level transformations will be applied.
These transformations include operation decomposition, [constant propagation](ConstPropagationPass.md),
shape inference, and canonicalization. 
* Then the ONNX dialect is [lowered to Krnl dialect](LoweringCode.md). 
To help debugging and performance tuning, onnx-mlir supports [instrumentation](Instrumentation.md)
at the ONNX operand level.
* All the passes may be controlled with [options](Options.md).
* How to handle errors can be found [here](ErrorHandling.md).
* How to support a new accelerator can be found [here](AddCustomAccelerators).

# Execution
The compiled ONNX model can be executed with either a
[C/C++ driver](mnist_example/README.md#write-a-c-driver-code)
[python driver](mnist_example/README.md#write-a-python-driver-code). or a
[java driver](mnist_example/README.md#write-a-java-driver-code).
The routine testing for onnx-mlir build is describe in this [document](Testing.md).
