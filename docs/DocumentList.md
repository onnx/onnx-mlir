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
* [UpdatingLLVMCommit.md](UpdatingLLVMCommit.md) describes how to update the commit of LLVM that ONNX-MLIR depends on.

# Development
* Onnx operation are represented with  [ONNX dialect](Dialects/onnx.md) in onnx-mlir.
* This [document](ImportONNXDefs.md#add_operation)
tell you how to generate an ONNX operation into ONNX dialect.
* After an ONNX model is imported into onnx-mlir, several graph-level transformations will be applied.
These transformations include operation decomposition, [constant propagation](ConstPropagationPass.md),
shape inference, and canonicalization. 
* Then the ONNX dialect is [lowered to Krnl dialect](LoweringCode.md). 
To help debugging and performance tuning, onnx-mlir supports [instrumentation](Instrumentation.md)
at the ONNX operand level.
* All the passes may be controlled with [options](Options.md).
* How to handle errors can be found [here](ErrorHandling.md).
* How to support a new accelerator can be found [here](AddCustomAccelerators.md).
* How to analyze unknown dimensions and query their equality at compile time can be found [here](DynamicDimensionAnalysis.md).
* A Jenkins monitor job was setup to help with updating LLVM commit. It locates the next commit we can update to without breaking ONNX-MLIR, as well as the commit that will break ONNX-MLIR. You can see the commit(s) here: [s390x](https://www.onnxmlir.xyz/jenkins/job/LLVM-Watch-Docker-Build/LLVM_20Watch_20Report/), [ppc64le](https://www.onnxmlir.xyz/jenkinp/job/LLVM-Watch-Docker-Build/LLVM_20Watch_20Report/), [amd64](https://www.onnxmlir.xyz/jenkinx/job/LLVM-Watch-Docker-Build/LLVM_20Watch_20Report/).

[#](#) Execution
The compiled ONNX model can be executed with either a
[C/C++ driver](mnist_example/README.md#write-a-c-driver-code)
[python driver](mnist_example/README.md#write-a-python-driver-code). or a
[java driver](mnist_example/README.md#write-a-java-driver-code).
The routine testing for onnx-mlir build is describe in this [document](Testing.md).
