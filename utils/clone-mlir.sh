git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout d421f5226048e4a5d88aab157d0f4d434c43f208 && cd ..
