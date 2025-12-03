<!--- SPDX-License-Identifier: Apache-2.0 -->
# onnx-mlir-python-driver
This light weight python driver for onnx-mlir compiler is a python package that does not depend on the building of onnx-mlir compiler or llvm-project. It uses  onnx-mlir compiler container through docker or podman python package, or locally installed onnx-mlir compiler to compile a model, and then run the compiled model through python interface. 
A simple example of using container can be found in [use_compiler_container.py](https://github.ibm.com/chentong/onnx-mlir-python-driver/blob/main/tests/use_compiler_container.py):
```
import numpy as np
import onnxmlir

a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

sess = onnxmlir.InferenceSession(
    "test_add.onnx",
    compile_args="-O3 --parallel",
    container_engine="docker",
    compiler_image_name="ghcr.io/onnxmlir/onnx-mlir-dev",
    compiler_path="/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
)
# In this example, all the options for InferenceSession related to compiler
# container are of the default value.
# You can simply use the following sentence.
"""
 sess = onnxmlir.InferenceSession("test_add.onnx", compile_args="-O3 --parallel")
"""
r = sess.run([a, b])
print(r)
```
This test case can be run with `python3 use_compiler_container.py` in the test directory.
## Installation
### Get the source code from git
```
git clone git@github.ibm.com:chentong/onnx-mlir-python-driver.git
```
### Make sure you are allowed to install python package
If your default environment does not allow to install python pakcage, you can use python virtual env. Here is the [reference](https://docs.python.org/3/library/venv.html):
```
# Create the virtual env
python3 -m venv path/to/store/your/installation
# Activate the virtual env
. path/to/store/your/installation/bin/active
```
### Install the package
If you want to use docker package:
```
pip3 install onnx-mlir-python-driver[docker]
```
If you want to use podman package:
```
pip3 install onnx-mlir-python-driver[podman]
```
### Verify
Run a test case in onnx-mlir-python-driver/tests. 
You can try the precompiled model first to just check the package with container:
```
cd onnx-mlir-python-driver/tests
python3 helloworld_with_precompiled_model.py
```
When you try the onnx-mlir container for the first time on your machine, it may take a while to pull the container from repo.
You can find more examples [here](https://github.ibm.com/chentong/onnx-mlir-python-driver/tree/main/tests).
