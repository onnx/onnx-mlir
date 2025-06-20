#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

##################### run_like_ort.py.py #######################################
#
# Copyright 2021-2025 The IBM Research Authors.
#
################################################################################
# Test case for the run_ort interface
################################################################################
import numpy as np
import onnxmlir

# Prepare input data
a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

sess = onnxmlir.InferenceSession("test_add.onnx", compile_args="-O3 --parallel")
r = sess.run_ort(["my_out"], {"x": a, "y": b})
print(r)
