git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout af20aff35ec37ead88903bc3e44f6a81c5c9ca4e && cd ..
