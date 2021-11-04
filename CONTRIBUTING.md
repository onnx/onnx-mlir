<!--- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to the ONNX-MLIR project

## Building ONNX-MLIR

Up to date info on how to build the project is located in the top directory [here](README.md). 

Since you are interested in contributing code, you may look [here](docs/Workflow.md) for detailed step by step directives on how to create a fork, compile it, and then push your changes for review.

## Guides for code generation for ONNX operations
* A guide on how to add support for a new operation is found [here](docs/HowToAddAnOperation.md).
* A guide to use Dialect builder details how to generate Krnl, Affine, MemRef, and Standard Dialect operations [here](docs/LoweringCode.md).
* A guide on how to best report errors is detailed [here](docs/ErrorHandling.md).
* Our ONNX dialect is derived from the machine readable ONNX specs. When upgrading the supported opset, or simply adding features to the ONNX dialects such as new verifiers, constant folding, canonicalization, or other such features, we need to regenerate the ONNX tablegen files. See [here](docs/ImportONNXDefs.md#how-to-use-the-script)) on how to proceed in such cases.
* To add an option to the onnx-mlir command, see instructions [here](docs/Options.md).
* To test new code, see [here](docs/Testing.md) for instructions.

## ONNX-MLIR specific dialects

* The onnx-mlir project is based on the opset version defined [here](docs/Dialects/onnx.md). This is a reference to a possibly older version of the current version of the ONNX operators defined in the onnx/onnx repo [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
* The Krnl Dialect is used to lower ONNX operators to MLIR affine. The Krnl Dialect is defined [here](docs/Dialects/krnl.md).
* To update the internal documentation on our dialects when there are changes, please look for guidance [here](docs/HowToAddAnOperation.md#update-your-operations-status).

## Testing and debugging ONNX-MLIR

* To test new code, see [here](docs/Testing.md) for instructions.
* We have support on how to trace performance issue using instrumentation. Details are found [here](docs/Instrumentation.md).
* We have support to debug numerical errors. See [here](docs/DebuggingNumericalError.md).

## Running ONNX models in Python and C

* Here is how to run a compiled model in python [link](docs/UsingPyRuntime.md).
* Here is the C runtime API to run models in C/C++ [link](http://onnx.ai/onnx-mlir/doxygen_html/OnnxMlirRuntime/index.html).

## Documentation

* To add documentation to our https://onnx.ai/onnx-mlir/, refer to instructions [here](docs/Documentation.md).

## Coordinating support for new ONNX operations

* Check this issue for status on operations required for ONNX Model Zoo [Issue 128](https://github.com/onnx/onnx-mlir/issues/128).
* Claim an op that you are working on by adding a comment on this [Issue #922](https://github.com/onnx/onnx-mlir/issues/922).

