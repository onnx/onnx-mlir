git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 28b27c1b10ae8d1f5b4fb9df691e8cf0da9be3f6 && cd ..
