#!/usr/bin/env python3

import numpy as np
from PyCompile import PyOMCompileSession

# Load onnx model and create PyOMCompileSession object.
file = './mnist.onnx'
compiler = PyOMCompileSession(file)
# Generate the library file. Success when rc == 0 while set the opt as "-O3"
rc = compiler.compile("-O3")
# Get the output file name
model = compiler.get_compiled_file_name()
if rc:
    print("Failed to compile with error code", rc)
    exit(1)
print("Compiled onnx file", file, "to", model, "with rc", rc)