git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 5e5a22caf88ac1ccfa8dc5720295fdeba0ad9372 && cd ..
