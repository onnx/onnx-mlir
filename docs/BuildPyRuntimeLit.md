# How to build and use PyRuntime lit

## Purpsoe

PyRuntime lit is a different way to build the original PyRuntime (src/Runtime/python).
All necessary dependence, such as llvm_project and onnx-mlir compiler is removed. The purpose is to easily build the python driver for the model execution on 
different systems. Currently, only the OMTenserUtils (src/Runtime), Python driver (src/Runtime/python), third_party/onnx and third_party/pybind11 are built.

The build of PyRuntime lit is controlled by a CMake option: ONNX_MLIR_ENABLE_PYRUNTIME_LIT. Without this option to cmake, the whole system remains the same.

## Functionalities
1. Build the python driver without llvm_project and onnx-mlir compiler built.
2. The python driver can be used with utils/RunONNXModel.py, or onnxmlir python package.
3. With PyRuntime lit, the compiler has not been built locally and docker image of onnx-mlir has to be usd to compile the model. The onnxmlir package contains
the python code to use python docker package to perform the compilation. Alternatively, the old script, onnx-mlir/docker/onnx-mlir.py, can do the fulfill the same task with subprocess and docker CLI.

## How to use
You can find the script for build and run at "onnx-mlir/utils/build-pyruntime-lit.sh.
```
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
```
