# How to build and use PyRuntime lit

## Purpsoe

onnx-mlir compiler can compile an ONNX model into a shared library (.so), and also provide a python driver to run the generated shared library through python code. That is PyRuntimeC. Originally, the PyRuntimeC is built with the onnx-mlir compiler, and consequently need the build of llvm_project. The PyRuntimeC lit is a different way to build the python d PyRuntime (src/Runtime/python) that does not need to
build llvm_project or onnx-mlir compiler. The purpose is to easily build the python driver for the model execution on different systems. The compilation can be done either with a local compiler, or with compiler container image through docker or podman package.
Currently, only the OMTenserUtils (src/Runtime), Python driver (src/Runtime/python), third_party/onnx and third_party/pybind11 are built.
The build of PyRuntime lit is controlled by a CMake option: ONNX_MLIR_ENABLE_PYRUNTIME_LIT. Without this option to cmake, the whole system remains the same.

## How to use
First, you need to create a python virtual env to be able to install python package, depending on the setup on your machine. For example,
```
python -m venv path/to/store/your/venv
. path/to/store/your/venv/bin/activate
```
You can find the script to set up and run test  at "onnx-mlir/utils/build-pyruntime-lit.sh.

```
#!/bin/bash

# Assume you are in an empty directory for build in cloned onnx-mlir.
# Usually it is "your_path/onnx-mlir/build"
# then you can run this script as "../util/build-pyruntime-lit.sh"

cmake .. -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON
make
make OMCreatePyRuntimePackage

# Install the package
pip3 install -e src/Runtime/python/onnxmlir
# -e is necessary for current package.
# To make sure to have container related package, you can install with
# pip3 install -e src/Runtime/python/onnxmlir[docker], or
# pip3 install -e src/Runtime/python/onnxmlir[podman]

# Run test case
cd src/Runtime/python/onnxmlir/tests
python3 test_1.py
# Current limitation on where the model is
```

If you already has the source code for onnx-mlir, you may simply create another build directory for PyRuntime light. For example:
```
cd path/to/existing/onnx-mlir
mkdir build-light
cd build-ligth
cmake .. -DONNX_MLIR_ENABLE_PYRUNTIME_LIT=ON
make
make OMCreatePyRuntimePackage

```
Note: the virtual env is not needed for compiler build, but only for pip install.
