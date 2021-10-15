<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX-MLIR Documentation

## Building

Up to date info on how to build the project is located in the top directory [here](../README.md).

## Guides for code generation
* A guide on how to add support for a new operation is found [here](HowToAddAnOperation.md).
* A guide to use Dialect builder details how to generate Krnl, Affine, MemRef, and Standard Dialect operations (to be updated).
* A guide on how to best report errors is detailed [here](ErrorHandling.md).
* Our ONNX dialect is derived from the machine readable ONNX specs. When upgrading the supported opset, or simply adding features to the ONNX dialects such as new verifiers, constant folding, canonicalization, or other such features, we need to regenerate the ONNX tablegen files. See [here](ImportONNXDefs.md) on how to proceed in such cases.
* To add an option to the onnx-mlir command, see instructions [here](Options.md).
* To test new code, see [here](Testing.md) for instructions.

## ONNX-MLIR specific dialects

* The onnx-mlir project is based on the opset version defined [here](Dialects/onnx.md). This is a reference to a possibly older version of the current version of the ONNX operators defined in the onnx/onnx repo [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
* The Krnl Dialect is used to lower ONNX operators to MLIR affine. The Krnl Dialect is defined [here](Dialects/krnl.md).

## Testing and debugging

* To test new code, see [here](Testing.md) for instructions.
* We have support on how to trace performance issue using instrumentation. Details are found [here](Instrumentation.md).
* We have support to debug numerical errors. See [here](DebuggingNumericalError.md).

## Running models in Python adn C

* Here is how to run a compiled model in python [link](UsingPyRuntime.md).
* Here is the C runtime API to run models in C/C++ [link](http://onnx.ai/onnx-mlir/doxygen_html/OnnxMlirRuntime/index.html).

## Documentation

* To add documentation to our https://onnx.ai/onnx-mlir/, refer to instructions [here](Documentation.md).

