#!/bin/bash
# first install MLIR in llvm-project
git clone https://DLCadmin:af52950e231dac7157c3d78d263d7468375fb102@github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX MLIR.
cd llvm-project && git checkout 076475713c236081a3247a53e9dbab9043c3eac2 && cd ..
