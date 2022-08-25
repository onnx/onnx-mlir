git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 59548fe873d8d98e359fb21fbb2a0852fed17ff5 && cd ..
