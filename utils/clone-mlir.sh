git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout d20ac95e9adf50fb589cf2187ec92abcedf27115 && cd ..
