git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 1053047a4be7d1fece3adaf5e7597f838058c947 && cd ..
