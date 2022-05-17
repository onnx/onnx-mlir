git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 9778ec057cf4214241e4a970f3e764e3cf150181 && cd ..
