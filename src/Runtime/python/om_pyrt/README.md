<!--- SPDX-License-Identifier: Apache-2.0 -->
# om_pyrt

## Functionalities
### Inference driver
This package provides a python driver to run inference on ONNX model compiled onnx-mlir.
There is a helloworld example in the tests folder with the package:
```
import numpy as np
import om_pyrt

# Initialize the inference session
# The onnx model simply performs tensor add on two 3x4x5xf32 tensors
# It is compiled into test_add.so with onnx-mlir
sess = om_pyrt.InferenceSession("./test_add.so")

# Prepare the inputs
a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

# Run inference
r = sess.run([a, b])

# Print output
print(r)
```
### Utilities to run a model

The `InferenceSession` provides the basic function to run a model. This package also provides some utility functions to run a model and verify the results.
The following example shows how to use the input files and reference output files (.npy files) to run a model. 

```
import numpy
import om_pyrt

args = om_pyrt.parse_args()
input_files = ["filename1.npy", "filename2.npy"]
ref_output_files = [ "ref_filename.npy" ]

session = om_pyrt.InferenceSession("mymode.so")
outputs = om_pyrt.run_model_with_file(session, input_files, ref_output_files, warmup=args.warmup, repeat=args.n_iteration, atol=args.atol, rtol=args.rtol)

```

## Install the pacakge
Currently, you need to build the package for your env and then install the package with pip.
In the env to run inference (no need to be the container for compiler)
1. Make sure that you are allowed to install python package. A common solution is to use python virtual environment
2. Get onnx-mlir source code when needed. Refer to [doc](https://github.com/onnx/onnx-mlir/blob/7423b55476bdf082cf3cb9a1bde9607d05de2992/docs/BuildOnLinuxOSX.md?plain=1#L58)
3. Create a build directory, e.g. build-om-pyrt. Then execute the following commands:

```
cd build-om-pyrt
cmake -DONNX_MLIR_TARGET_TO_BUILD=OMPyRt ..
cmake --build . --target OMCreateOMPyRtPackage
pip install -e src/Runtime/python/om_pyrt
```

## Compile onnx model to shared library
TBD


## Test
Suppose you are in build-om-pyrt directory:

```
cd src/Runtime/python/om_pyrt/tests
# Compile test_add.mlir into test_add.so
python helloworld.py
```
