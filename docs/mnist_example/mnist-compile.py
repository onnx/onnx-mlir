#!/usr/bin/env python3

import numpy as np
from PyOMCompile import OMCompile

# Load onnx model and create OMCompile object.
if True:
    # model name given separately, works.
    model = "./mnist.onnx"
    flags = "-O3 -o mnist"
if False:
    # model name given in flags, works.
    model = ""
    flags = "-O3 -o mnist ./mnist.onnx"
if False:
    # no model, fails.
    model = ""
    flags = "-O3"

try:
    compiler = OMCompile()
    compiled_model = compiler.compile(model, flags)
except RuntimeError as e:
    print(f"Compilation failed: {e}")
    exit(1)
print("Compiled onnx file", model, "to", compiled_model)
