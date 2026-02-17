<!--- © Copyright IBM Corp. 2025 -->
# OMPyInfer
This package provides a python driver to run inference on ONNX model compiled withe IBM Z Deep Learning Compiler (zDLC).
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
To compile an onnx model to a shared library download the [IBM zDLC image](https://ibm.github.io/ibm-z-oss-hub/containers/zdlc.html) and follow the instructions outlined for using [IBM zDLC](https://github.com/IBM/zDLC/tree/main). 

## Pre-requisites for OMPyInfer
These pre-requisities are currently provided as part of the OMPyInfer package for Python versions 3.9 until 3.13. In case for any other python version you can copy them from the zDLC container.

Follow these instructions when there is a different python version than the ones listed above.To get the libs needed to run inference on ONNX model compiled withe IBM Z Deep Learning Compiler (zDLC) container, one can copy them from the IBM Z Deep Learning Compiler container.

Login into the icr container registry 
```
podman login -u iamapikey icr.io 
Password:  <- Paste your apikey here
Login Succeeded

```

Pull the zdlc container image from icr.io :
```
podman pull icr.io/ibmz/zdlc:5.0.0

```

Copy the libs from the zdlc container
```
podman run --rm -v ${pwd}:/files:z --entrypoint '/usr/bin/bash' icr.io/ibmz/zdlc:5.0.0 -c "cp /usr/local/lib/PyRuntime* /files"

```
Once the libs are copied out of the container, one can move them to the `src/OMPyInfer/libs` folder like the current libs.

## Install
You can install directly from IBM github with the following command:
```
pip install git+ssh://git@github.ibm.com/zosdev/OMPyInfer.git
```

Or you can install from your local disk after cloning the repo:
```
git clone git@github.ibm.com:zosdev/OMPyInfer.git
pip install ./OMPyInfer
```

## Verify
Clone the repo and run the test example.
```
git clone git@github.ibm.com:zosdev/OMPyInfer.git
cd OMPyInfer/tests
python helloworld.py
```


