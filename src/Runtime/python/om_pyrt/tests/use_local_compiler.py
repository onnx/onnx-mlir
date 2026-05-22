#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################# use_local_compiler.py  #######################################
#
# Copyright 2021-2026 The IBM Research Authors.
#
################################################################################
# Test case to use the local compiler to compile the model
################################################################################

# Local model file
from pathlib import Path

script_dir = Path(__file__).resolve().parent
model_file = str(script_dir / "test_add.mlir")

# The compiler_path is used to locate the compiler.
# compile_args is the flags passed to onnx-mlir
import om_pyrt

try:
    compile_session = om_pyrt.CompileSession(
        compiler_path="/Users/chentong/Projects/onnx-mlir/build/Debug/bin/onnx-mlir"
    )
    compile_session.compile(model_file, "-O3")
except Exception as e:
    print("Fialed to compile")
    exit(-1)

compiled_model = compile_session.get_output_file_name()

# Prepare input data
import numpy as np

a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

# Run inference
sess = om_pyrt.InferenceSession(compiled_model)
r = sess.run([a, b])
print(r)
