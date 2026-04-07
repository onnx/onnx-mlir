# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import numpy as np
import OMPyCompile

# Initialize the inference session
# The onnx model simply performs tensor add on two 3x4x5xf32 tensors
# The test_add.so is compiled by onnx-mlir from test_add.onnx
# You need to create your own test_add.so from test_add.onnx or test_add.mlir
# unless you are on a s390 machine.
script_dir = Path(__file__).resolve().parent
model = script_dir / "test_add.mlir"
r = OMPyCompile.compile(str(model), "-O3")

print(r)
