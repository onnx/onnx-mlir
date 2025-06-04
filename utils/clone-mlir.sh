git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 06faded057ff77737db42468be3c6e18b0163332 && cd ..
