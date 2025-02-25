git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 43d71baae36c8d8b5a9995aa35efebe09cc9c2d6 && cd ..
