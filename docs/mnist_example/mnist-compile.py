#!/usr/bin/env python3

import numpy as np
from PyRuntime import ExecutionSession
from PyOnnxMlirCompiler import OnnxMlirCompiler, OnnxMlirTarget, OnnxMlirOption

# Load onnx model and create Onnx Mlir Compiler object.
# Compiler needs to know where to find its runtime. Set ONNX_MLIR_RUNTIME_DIR
# to proper path.
# export ONNX_MLIR_RUNTIME_DIR=../../build/Debug/lib
file = './mnist.onnx'
compiler = OnnxMlirCompiler(file)
# Set optimization level to -O3.
compiler.set_option(OnnxMlirOption.opt_level, "3")
#compiler.set_option(OnnxMlirOption.verbose, "")
print("Compile", file, "with opt level", compiler.get_option(OnnxMlirOption.opt_level), "for arch", compiler.get_option(OnnxMlirOption.target_arch), "and CPU", compiler.get_option(OnnxMlirOption.target_cpu))
# Generate the library file. Success when rc == 0.
rc = compiler.compile('./mnist', OnnxMlirTarget.emit_lib)
model = compiler.get_output_file_name()
if rc:
    print("Failed to compile with error code", rc)
    exit(1)
print("Compiled onnx file", file, "to", model, "with rc", rc)