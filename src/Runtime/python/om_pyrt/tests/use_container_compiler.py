#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

############# use_compiler_container.py #######################################
#
# Copyright 2021-2026 The IBM Research Authors.
#
################################################################################
# Test case to compile a model with compiler container
################################################################################

# Local model file
from pathlib import Path

script_dir = Path(__file__).resolve().parent
model_file = str(script_dir / "test_add.mlir")

# To use compiler container, the image name and compiler path in the image
# need to be provided.
# compile_args is the flags passed to onnx-mlir
import om_pyrt

try:
    compile_session = om_pyrt.CompileSession(
        compiler_image="ghcr.io/onnxmlir/onnx-mlir-dev:s390x",
        compiler_path="/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
    )

    compiled_model = compile_session.compile(
        model_file, "-O3", reuse_compiled_model=True
    )
except Exception as e:
    print("Failed to compile")
    exit(-1)

# Prepare input data
import numpy as np

a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

# Run inference
sess = om_pyrt.InferenceSession(compiled_model)
r = sess.run([a, b])
print(r)
