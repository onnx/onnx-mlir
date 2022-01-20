<!--- SPDX-License-Identifier: Apache-2.0 -->

# Define and Use Command-line Options for ONNX-MLIR

Command-line options can be used to alter the default behavior of onnx-mlir, or onnx-mlir-opt, and help user experimenting, debugging or performance tuning. We implemented command-line in ONNX-MLIR based on the command-line utility provided by LLVM. We did not define `Option` or `ListOption` with MLIR pass classes(see discussion). 
 
## Organize Options
Refer [llvm document](https://llvm.org/docs/CommandLine.html) for basic idea of how to define an option. In ONNX-MLIR, options are put into groups (`llvm::cl::OptionCategory`).
One group of options are only used by onnx-mlir to configure its input or output. These options are defined in src/main.cpp and src/Compiler/CompilerUtils.cpp within OnnxMlirOptions category.
The rest of options may be used by both onnx-mlir and onnx-mlir-opt to control the behavior of a pass or passes. So far, only one group is defined as an example. 

## Code structure
The head file for options is `src/Support/OMOptions.hpp`. This file should contain the declaration of groups used by both onnx-mlir and onnx-mlir-opt, and options that may be shared by different passes.
The definition of group and shared options are in `src/Support/OMOptions.cpp`.

## Define an option
* Add a declaration of the option in 'src/Support/OMOptions.hpp`
* In `src/SupportOMOptions.cpp`, define the option with the desirable llvm::cl class and option group
* For the file to use this option, make sure `src/Support/OMOptions.hpp` is included.
* Add OMSupport to the list of libraries in CMakefile.txt

## Define an option local to a transformation
If an option is only used in one transformation,  it can be defined locally in the file for the transformation. In the file for transformation:
* Include `src/Support/OMOtions.hpp`
* Define the option with the desirable llvm::cl class and option group
* Add OMSupport to the list of libraries in CMakefile.txt

## Discussion
### MLIR Option support
MLIR allows to define options within the pass class definition. However, the document said you have to define the same options for PassPipeline and pass a lambda function to assign the options. I thought that this requirement complicated code. I am open to using MLIR if someone know how to easily implement it and know its advantages.
