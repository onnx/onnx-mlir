git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 52e1d522a65b0d30cf1d49851d2ed6d196e65e10 && cd ..
