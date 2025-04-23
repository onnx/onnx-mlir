git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 365c2d67ebfa8710610ebf8689f1f1bed54f43b1 && cd ..
