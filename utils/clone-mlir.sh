git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 721be578eada1b787dab2996e483bd220b92f567 && cd ..
