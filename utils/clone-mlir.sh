git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 029313cc979ae71877b65794b1063d4e51184cc8 && cd ..
