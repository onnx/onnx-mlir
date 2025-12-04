git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 42a8ff877d47131ecb1280a1cc7e5e3c3bca6952 && cd ..
