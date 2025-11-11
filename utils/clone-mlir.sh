git clone -n https://github.com/xilinx/llvm-aie.git llvm-project
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 081636148c86 && cd ..
