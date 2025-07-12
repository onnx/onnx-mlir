git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 7a33569510535f0b917a2e50f644bf57490aee24 && cd ..
