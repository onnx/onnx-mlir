<!--- SPDX-License-Identifier: Apache-2.0 -->
# OMPyInfer
This package provides a python driver to run inference on ONNX model compiled onnx-mlir.
There is a helloworld example in the tests folder with the package:
```
import numpy as np
import OMPyInfer

# Initialize the inference session
# The onnx model simply performs tensor add on two 3x4x5xf32 tensors
# It is compiled into test_add.so with zDLC
sess = OMPyInfer.InferenceSession("./test_add.so")

# Prepare the inputs
a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

# Run inference
r = sess.run([a, b])

# Print output
print(r)
```
## Utilities to run a model

The `InferenceSession` provides the basic function to run a model. This package also provides some utility functions to run a model and verify the results.
The following example shows how to use the input files and reference output files (.npy files) to run a model. 

```
import numpy
import OMPyInfer

input_files = ["filename1.npy", "filename2.npy"]
ref_output_files = [ "ref_filename.npy" ]

session = OMPyInfer.InferenceSession("mymode.so")
outputs = OMPyInfer.utils.run_model_with_file(session, input_files, ref_output_files, rtol, atol)

```

## Compile onnx model to shared library
TBD


## Pre-requisites for OMPyInfer
These pre-requisities are currently provided as part of the OMPyInfer package for Python versions 3.9 until 3.13. 
Prebuilt libraries for Linux on Z is provided.
Follow these instructions (TBD) to build the libraries for your own system.

## Install
Currently, only local installation is supported.
Suppose you have onnx-mlir cloned on your machine. Install OMPyInfer with the following command:
```
python onnx-mlir/src/Runtime/python/OMPyInfer
```


## Test
```
cd OMPyInfer/tests
python helloworld.py
```
