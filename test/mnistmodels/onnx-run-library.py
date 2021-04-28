#!/usr/bin/env python3

import onnx
import os
import sys
import argparse
from onnx import numpy_helper
import numpy as np

# Reference backend, use onnxruntime by default
import onnxruntime

try:
    from PyRuntime import ExecutionSession
except ImportError:
    raise ImportError(
        "Unable to import from PyRuntime. Build the PyRuntime target and make sure that the library is in your search path (or PYTHONPATH)."
    )

parser = argparse.ArgumentParser()
parser.add_argument('model_lib')
parser.add_argument('test_data_dir')
args = parser.parse_args()

model_abs_path = os.path.abspath(args.model_lib)
test_data_dir = os.path.abspath(args.test_data_dir)
sess = ExecutionSession(model_abs_path,
                        "run_main_graph")

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