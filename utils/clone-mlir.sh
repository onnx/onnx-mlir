git clone -n https://github.com/xilinx/llvm-aie.git llvm-project
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 60a1abaabb5721b7ee6f1a085072f9ad7e7bc430 && cd ..
