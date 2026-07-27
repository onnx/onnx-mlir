<!--- SPDX-License-Identifier: Apache-2.0 -->

# Index of documents
This document serves as an index for onnx-mlir documents.

# About
* [ONNXAI.md](ONNXAI.md) gives a short introduction to the project and how to get in touch (Slack channel).

# Installation and working environment
* [Prerequisite.md](Prerequisite.md) lists the software required to build onnx-mlir.
* Installation is covered by [README.md](../README.md), with OS-specific details in [BuildOnLinuxOSX.md](BuildOnLinuxOSX.md) and [BuildOnWindows.md](BuildOnWindows.md).
* [BuildONNX.md](BuildONNX.md) describes how to install `third_party ONNX` for backend tests or to regenerate ONNX operations.
* [BuildStandalone.md](BuildStandalone.md) describes how to build onnx-mlir as a standalone binary, without a full LLVM/MLIR build tree.
* [Docker.md](Docker.md) describes how to build and develop onnx-mlir using Docker, and [DockerInDocker.md](DockerInDocker.md) covers Docker-in-Docker support.
* [Workflow.md](Workflow.md) describes how to contribute in the github environment.
* [This guideline](Documentation.md) is used to keep documentation and code consistent.
* [UpdatingLLVMCommit.md](UpdatingLLVMCommit.md) describes how to update the commit of LLVM that onnx-mlir depends on.
* A Jenkins monitor job was setup to help with updating LLVM commit. It locates the next commit we can update to without breaking ONNX-MLIR, as well as the commit that will break ONNX-MLIR. You can see the commit(s) here: [s390x](https://www.onnxmlir.xyz/jenkins/job/LLVM-Watch-Docker-Build/LLVM_20Watch_20Report/), [ppc64le](https://www.onnxmlir.xyz/jenkinp/job/LLVM-Watch-Docker-Build/LLVM_20Watch_20Report/), [amd64](https://www.onnxmlir.xyz/jenkinx/job/LLVM-Watch-Docker-Build/LLVM_20Watch_20Report/).

# Supported ONNX Ops
* CPU support is covered [here](SupportedONNXOps-cpu.md).
* NNPA support is covered [here](SupportedONNXOps-NNPA.md), with additional operations covered in [SupportedOps-NNPA-supplement.md](SupportedOps-NNPA-supplement.md).

# Development
* Onnx operations are represented with the [ONNX dialect](Dialects/onnx.md) in onnx-mlir. Related dialects used in lowering are [Krnl](Dialects/krnl.md), [ZHigh](Dialects/zhigh.md), and [ZLow](Dialects/zlow.md).
* This [document](ImportONNXDefs.md#add_operation)
tells you how to generate an ONNX operation into the ONNX dialect.
* After an ONNX model is imported into onnx-mlir, several graph-level transformations will be applied.
These transformations include operation decomposition, [constant propagation](ConstPropagationPass.md),
shape inference, and canonicalization.
* Then the ONNX dialect is [lowered to Krnl dialect](LoweringCode.md). Documents describing specific
lowering/optimization strategies live under [optimization-onnx-lowering/](optimization-onnx-lowering/):
[adding a fusion pattern](optimization-onnx-lowering/AddingAFusionPattern.md),
[Conv lowering with Im2Col and MatMul](optimization-onnx-lowering/ConvIm2Col.md),
[ConvTranspose decomposition](optimization-onnx-lowering/ConvTranspose.md),
[ConvTranspose with output_shape support](optimization-onnx-lowering/ConvTranspose-OutputShape.md), and
[GridSample bilinear 2D optimization](optimization-onnx-lowering/GridSample.md).
To help debugging and performance tuning, onnx-mlir supports [instrumentation](Instrumentation.md)
at the ONNX operand level.
* All the passes may be controlled with [options](Options.md).
* How to handle errors can be found [here](ErrorHandling.md).
* How to support a new accelerator can be found [here](AddCustomAccelerators.md).
* How to analyze unknown dimensions and query their equality at compile time can be found [here](DynamicDimensionAnalysis.md).
* How location info is maintained and used during transformation and debugging is covered in [LocationInfo.md](LocationInfo.md).
* How onnx-mlir handles the ONNX Sequence type is covered in [SequenceType.md](SequenceType.md).

# NNPA Accelerator
* [AccelNNPAHowToUseAndTest.md](AccelNNPAHowToUseAndTest.md) describes how to build and test onnx-mlir with the NNPA accelerator.
* [Quantization-NNPA.md](Quantization-NNPA.md) describes quantization support on NNPA.
* [JsonConfigFile-NNPA.md](JsonConfigFile-NNPA.md) describes the NNPA-specific JSON configuration options for device placement and quantization.

# Execution
The compiled ONNX model can be executed with either a
[C/C++ driver](mnist_example/README.md#write-a-c-driver-code),
[python driver](mnist_example/README.md#write-a-python-driver-code), or a
[java driver](mnist_example/README.md#write-a-java-driver-code).
* [UsingPyRuntime.md](UsingPyRuntime.md) describes the Python runtime interfaces.
* [BuildPyRuntimeLight.md](BuildPyRuntimeLight.md) describes building and using PyRuntimeC in lightweight mode.
* [PythonPackage.md](PythonPackage.md) describes the installable `onnxmlir` Python package.
* [RunTorchModel.md](RunTorchModel.md) describes installing the `torch_onnxmlir` package to run a torch model via `torch.compile()`.
* [JsonConfigFile.md](JsonConfigFile.md) describes the general JSON configuration file for specifying compile options.

# Testing and Debugging
The routine testing for onnx-mlir build is described in this [document](Testing.md).
* [TestingHighLevel.md](TestingHighLevel.md) covers build trouble-shooting and higher-level testing.
* [DebuggingNumericalError.md](DebuggingNumericalError.md) describes how to debug numerical errors between onnx-mlir and a reference implementation.
* [ProfileModel.md](ProfileModel.md) describes profiling a compiled ONNX model with `utils/profile-model.py`.
* [PerformanceTesting.md](PerformanceTesting.md) describes gathering and analyzing runtime/compile-time performance statistics with `RunONNXModel.py`, `--profile-ir`/`--profile-ir-with-sig`, and `utils/make-report.py`.
