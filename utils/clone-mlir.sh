git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 21f4b84c456b471cc52016cf360e14d45f7f2960 && cd ..
