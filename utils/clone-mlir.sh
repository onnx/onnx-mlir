git clone -n https://github.com/xilinx/llvm-aie.git llvm-project
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 776b07b472a12db1a451fb4bfc737e05c0ee0b1c && cd ..
