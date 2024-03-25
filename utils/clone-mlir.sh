git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout e5ed7b6e2fd368b722b6359556cd0125881e7638 && cd ..
