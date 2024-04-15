git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 1e6ce5e284f5c0e8d64eee21af727bb164eb3caf && cd ..
