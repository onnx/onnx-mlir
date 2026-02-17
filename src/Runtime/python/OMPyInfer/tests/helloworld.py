# SPDX-License-Identifier: Apache-2.0

import numpy as np
import OMPyInfer

# Initialize the inference session
# The onnx model simply performs tensor add on two 3x4x5xf32 tensors
# The test_add.so is compiled by onnx-mlir from test_add.onnx
sess = OMPyInfer.InferenceSession("./test_add.so")

# Prepare the inputs
a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

# Run inference
r = sess.run([a, b])

# Print output
print(r)
