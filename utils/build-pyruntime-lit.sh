#!/bin/bash

# Assume you are in an empty directory for build in cloned onnx-mlir.
# Usually it is "your_path/onnx-mlir/build"
# then you can run this script as "../util/build-pyruntime-lit.sh"

cmake .. -DONNX_MLIR_ENABLE_PYRUNTIME_LIT=ON
make
make OMCreatePyRuntimePackage

# Install the package
pip3 install src/Runtime/python/onnxmlir

# Run test case
python3 src/Runtime/python/onnxmlir/test/test_1.py
