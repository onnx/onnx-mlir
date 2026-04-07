<!--- SPDX-License-Identifier: Apache-2.0 -->
# OMPyInfer

## Functionalities
### Inference driver
This package provides a python driver to compile an ONNX model with standalone onnx-compile compiler (built in container, but executed outside of the container).
There is a helloworld example in the tests folder with the package:
```
import OMPyCompile

model="./test_add.so"

compiled_lib = OMPyCompile.compile(model, flags="-O3")

# Print output: the path to the compiled model
print(compile_lib)
```

## Create the pacakge
Refer to docs/BuildStandAlone.md to build the standalone compiler. Make sure to run "cmake --build . --target OMCreateOMPyCompilePackage" in your build-standalone directory.

## Install the package
In the env to use the standalone compiler (no need to be the container for compiler), make sure it is allowed to install python package. A common solution is to use python virtual environment

```
pip install  onnx-mlir/build-standalone/src/Runtime/python/OMPyCompile
```

## Test
Suppose you are in build-standalone directory:

```
cd src/Runtime/python/OMPyCompile/tests
python helloworld.py
ls test_add.so
```
