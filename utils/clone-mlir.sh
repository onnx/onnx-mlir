git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 91088978d712cd7b33610c59f69d87d5a39e3113 && cd ..
