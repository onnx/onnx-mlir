git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout c511c90680eecae2e4adb87f442f41d465feb0f2 && cd ..
