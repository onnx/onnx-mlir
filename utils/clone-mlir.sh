git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 8b76aea8d8b1b71f6220bc2845abc749f18a19b7 && cd ..
