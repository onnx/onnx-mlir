# How to build and use PyRuntimeC in a light way

## Overview

onnx-mlir compiler can compile an ONNX model into a shared library (.so), and also provide a python driver, PyRuntimeC, to run the generated shared library through python code. Originally, the PyRuntimeC is built with the onnx-mlir compiler, and consequently need the build of llvm_project.
Here a light way to build PyRuntimeC without llvm_project nor the other part of onnx-mlir compiler is provided.
Therefore, user can easily build the python driver for the model execution on different systems. The compilation can be done either with a local compiler, or with compiler container image through docker or podman package.
Currently, only the OMTenserUtils (src/Runtime), Python driver (src/Runtime/python), third_party/onnx and third_party/pybind11 are built.
The build of light PyRuntime is controlled by a CMake option: ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT. When this option is set to 'OFF' (the default value), there is no change to the build of onnx-mlir.

## How to build

Assume that you have cloned onnx-mlir source code, and is using `build` directory for you normal onnx-mlir compiler build. You need to create a new build directory for PyRuntimeC light, for example, build-light.
In the build-light directory, you can execute the following commands:
```
cmake .. -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON
make
make OMCreatePyRuntimePackage
```
Please refer to [script](../util/build-pyruntime-light.sh)


## How to install
First, you need to create a python virtual env to be able to install python package, depending on the setup on your machine. For example,
```
python -m venv path/to/store/your/venv
. path/to/store/your/venv/bin/activate
```

Then you can install the onnxmlir package that contains the python driver.
```
#!/bin/bash

# Install the package
pip3 install -e src/Runtime/python/onnxmlir[docker]
# The option -e is necessary for current package.
# To use podman, replace the previous command with the following one
# pip3 install -e src/Runtime/python/onnxmlir[podman]
```

## Run test case
```
cd src/Runtime/python/onnxmlir/tests
python3 helloworld_with_precompiled_model.py
```
