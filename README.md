# ONNF
Open Neural Network Frontend : an ONNX frontend to MLIR.

[![CircleCI](https://circleci.com/gh/clang-ykt/ONNF.svg?style=svg)](https://circleci.com/gh/clang-ykt/ONNF)

## Installation

We assume an existing installation of MLIR. The LLVM-Project repo commit hash we used to test against is 9b6ad8466bb8b97082b705270603ad7f4559e931 and the MLIR repo commit hash we used is 0710266d0f56cf6ab0f437badbd7416b6cecdf5f. 

Two environment variables need to be set:
- LLVM_SRC should point to the llvm src directory (e.g., llvm-project/llvm).
- LLVM_BUILD should point to the llvm build directory (e.g., llvm-project/build).

To build ONNF, use the following command:
```
git clone --recursive git@github.com:clang-ykt/ONNF.git
mkdir build
cd build
cmake ..
cmake --build . --target all
```

After the above commands succeed, an `onnf` executable should appear in the `bin` directory. 

## Using ONNF

The usage of `onnf` is as such:
```
OVERVIEW: ONNF MLIR modular optimizer driver

USAGE: onnf [options] <input file>

OPTIONS:

Generic Options:

  --help        - Display available options (--help-hidden for more)
  --help-list   - Display list of available options (--help-list-hidden for more)
  --version     - Display the version of this program

ONNF Options:
These are frontend options.

  Choose target to emit:
      --EmitONNXIR - Ingest ONNX and emit corresponding ONNX dialect.
      --EmitMLIR   - Lower model to MLIR built-in transformation dialect.
      --EmitLLVMIR - Lower model to LLVM IR (LLVM dialect).
      --EmitLLVMBC - Lower model to LLVM IR and emit (to file) LLVM bitcode for model.
```

## Example

For example, to lower an ONNX model (e.g., add.onnx) to ONNX dialect, use the following command:
```
./onnf --EmitONNXIR add.onnx
```
The output should look like:
```
module {
  func @main_graph(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    return %0 : tensor<10x10x10xf32>
  }
}
```

