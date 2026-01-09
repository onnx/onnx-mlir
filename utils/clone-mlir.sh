git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 0c2701fe7fa002e1befc5f86c268a7964f96d286 && cd ..
