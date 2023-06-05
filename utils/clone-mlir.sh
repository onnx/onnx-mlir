git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 6cf7fe4a9a715bcdf3f4913753109e22dfc9940b && cd ..
