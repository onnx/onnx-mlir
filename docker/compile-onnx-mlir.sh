#!/bin/bash
# Export environment variables pointing to LLVM-Projects.
export MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir

pwd
ls -al
mkdir onnx-mlir/build && cd onnx-mlir/build
cmake ..
cmake --build . --target onnx-mlir

# Run FileCheck tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
