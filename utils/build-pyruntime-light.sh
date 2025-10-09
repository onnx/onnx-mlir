#!/bin/bash

# Assume you are in an empty directory for build in cloned onnx-mlir.
# For example onnx-mlir/build-light, when onnx-mlir/build is used for ordinary
# compiler build

cmake .. -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON
make
make OMCreatePyRuntimePackage

# Install the package
pip3 install -e src/Runtime/python/onnxmlir[docker]
# The option -e is necessary for current package.
# To use podman, replace the previous command with the following one
# pip3 install -e src/Runtime/python/onnxmlir[podman]

# Run test case
cd src/Runtime/python/onnxmlir/tests
python3 helloworld_with_precompiled_model.py
