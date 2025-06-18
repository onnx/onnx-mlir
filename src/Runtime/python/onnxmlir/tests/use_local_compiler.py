#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################# use_local_compiler.py  #######################################
#
# Copyright 2021-2025 The IBM Research Authors.
#
################################################################################
# Test case to use the local compiler to compile the model
################################################################################
import numpy as np
import onnxmlir

# Prepare input data
a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

# When compiler_image_name is None, local compiler will be used. The compiler_path is needed to
# locate the compiler.
# Alternative implementation:  use env variable ONNX_MLIR_HOME?
# compile_args is the flags passed to onnx-mlir
sess = onnxmlir.InferenceSession(
    "test_add.onnx",
    compile_args="-O3",
    compiler_image_name=None,
    compiler_path="/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
)
r = sess.run([a, b])
print(r)
