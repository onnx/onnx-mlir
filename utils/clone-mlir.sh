git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout e8be3bea2ce0ec51b614cd7eb7d5d3a1e56d9524 && cd ..
