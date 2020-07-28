#!/usr/bin/env python3

import onnx
import os
import sys
import argparse
from onnx import numpy_helper
import numpy as np

# Reference backend, use onnxruntime by default
import onnxruntime

if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found.")

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(os.environ['ONNX_MLIR_HOME'], "lib")
sys.path.append(RUNTIME_DIR)

try:
    from PyRuntime import ExecutionSession
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyRuntime target, build it by running `make PyRuntime`."
    )

parser = argparse.ArgumentParser()
parser.add_argument('model_lib')
parser.add_argument('test_data_dir')
args = parser.parse_args()

model_abs_path = os.path.abspath(args.model_lib)
test_data_dir = os.path.abspath(args.test_data_dir)
sess = ExecutionSession(model_abs_path,
                        "_dyn_entry_point_main_graph")

inputs = []
input_file = os.path.join(test_data_dir, 'input_0.pb')
tensor = onnx.TensorProto()
with open(input_file, 'rb') as f:
    tensor.ParseFromString(f.read())
inputs.append(numpy_helper.to_array(tensor))

# Run inference with onnx-mlir
outputs = sess.run(inputs)
print("Predicted output: \n")
print(outputs)
print()

true_outputs = []
output_file = os.path.join(test_data_dir, 'output_0.pb')
output_tensor = onnx.TensorProto()
with open(output_file, 'rb') as f:
    output_tensor.ParseFromString(f.read())
true_outputs.append(numpy_helper.to_array(output_tensor))
print("True output: \n")
print(true_outputs)