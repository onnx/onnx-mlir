<!--- SPDX-License-Identifier: Apache-2.0 -->
# OMPyInfer
This package provides a python driver to run inference on ONNX model compiled onnx-mlir.
There is a helloworld example in the tests folder with the package:
```
# IBM Confidential
# © Copyright IBM Corp. 2025

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


## Verify
```
cd OMPyInfer/tests
python helloworld.py
```

## VERSIONS
Version 1.0.0 supports the model copied with onnx-mlir before 29bde823f, 2026-02-04.

