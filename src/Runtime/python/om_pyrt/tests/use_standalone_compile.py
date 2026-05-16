#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################# use_local_compiler.py  #######################################
#
# Copyright 2021-2025 The IBM Research Authors.
#
################################################################################
# Test case to use the local compiler to compile the model
################################################################################

# Local model file
from pathlib import Path

script_dir = Path(__file__).resolve().parent
model_file = str(script_dir / "test_add.mlir")

# When compiler_image_name is None, local compiler will be used.
# The compiler_path is used to locate the compiler.
# compile_args is the flags passed to onnx-mlir
import om_pyrt

compiled_model = om_pyrt.CompileWithStandalone(
    "./test_add.mlir",
    "-O3",
)
print(compiled_model)

# Prepare input data
import numpy as np

a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

# Run inference

sess = om_pyrt.InferenceSession(compiled_model)
r = sess.run([a, b])
print(r)
