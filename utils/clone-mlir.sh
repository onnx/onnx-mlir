git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 1fe0b8b29017849a35629b93da94992e1dbf2e10 && cd ..
