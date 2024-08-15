git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 1b038a97af56 && cd ..
