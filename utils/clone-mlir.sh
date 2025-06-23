git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 590a988c1ddf5dd7507f48b45a10bbf2dac84e01 && cd ..
