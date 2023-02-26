git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 5e111eb275eee3bec1123b4b85606328017e5ee5 && cd ..
