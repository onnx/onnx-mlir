git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 01d233ff403823389f8480897e41aea84ecbb3d3 && cd ..
