#!/bin/bash

# Assume you are in an empty directory for build in cloned onnx-mlir.
# Usually it is "your_path/onnx-mlir/build"
# then you can run this script as "../util/build-pyruntime-lit.sh"

cmake .. -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON
make
make OMCreatePyRuntimePackage

# Install the package
pip3 install -e src/Runtime/python/onnxmlir
# -e is necessary for current package. Need to add resource description
# to install the pre-compiled binary

# Run test case
cd src/Runtime/python/onnxmlir/tests
python3 test_1.py
# Current limitation on where the model is
