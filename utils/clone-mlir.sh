git clone -n https://github.com/xilinx/llvm-aie.git llvm-project
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 39149e4f236f7cb37d510e5d2445f86385ac3b6b && cd ..
