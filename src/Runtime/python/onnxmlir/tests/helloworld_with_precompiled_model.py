#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

############# helloword_with_precompiled_model.py  #############################
#
# Copyright 2021-2025 The IBM Research Authors.
#
################################################################################
# Test case to run the pre-compiled .so
################################################################################

import numpy as np
import onnxmlir

# Prepare input data
a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

sess = onnxmlir.InferenceSession("test_add.so")
r = sess.run([a, b])
print(r)
