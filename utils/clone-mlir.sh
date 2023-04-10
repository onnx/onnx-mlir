git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 76c83b3595a534c5b28bd80039e2115358ba2291 && cd ..
