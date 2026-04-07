<!--- SPDX-License-Identifier: Apache-2.0 -->
# OMPyCompiler

## Functionalities
### Compiler driver
This package provides a python driver to compile an ONNX model with standalone onnx-compile compiler (built in container, but executed outside of the container).
There is a helloworld example in the tests folder with the package:

```
import OMPyCompile

model="./test_add.onnx"

compile_session = OMPyCompile.OMCompile(model, "-O3")

r = compile_session.get_output_file_name()

# Print output: the path to the compiled model
print(r)
```

## Create the pacakge
### Build standalone compiler
Refer to docs/BuildStandAlone.md to build the standalone compiler. 
Assume that the path of the standalone compiler is onnx-mlir/build-standalone/Debug.
### Create and install the package
Refere to TBD (temporarily use README.md in of OMPyInfer package).
Suppose you are under onnx-mlir directory
```
mkdir build-OMPyInfer
cd build-OMPyInfer
cmake -DONNX_MLIR_TARGET_TO_BUILD=OMPyInfer -DONNX_MLIR_STANDALONE_DIR=onnx-mlir/build-standalone/Debug ..
cmake --build . --target OMCreateOMPyCompilePackage
pip install src/Runtime/python/OMPyCompile
```

## Test

```
cd onnx-mlir/src/Runtime/python/OMPyCompile/tests
python helloworld.py
ls test_add.so
```
