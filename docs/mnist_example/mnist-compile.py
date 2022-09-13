#!/usr/bin/env python3

import numpy as np
from PyOnnxMlirCompiler import OnnxMlirCompiler

# Load onnx model and create Onnx Mlir Compiler object.
file = './mnist.onnx'
compiler = OnnxMlirCompiler(file)
# Generate the library file. Success when rc == 0 while set the opt as "-O3"
rc = compiler.compile_from_file("-O3")
# Get the output file name
model = compiler.get_output_file_name()
if rc:
    print("Failed to compile with error code", rc)
    exit(1)
print("Compiled onnx file", file, "to", model, "with rc", rc)