git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 23aa5a744666b281af807b1f598f517bf0d597cb && cd ..
