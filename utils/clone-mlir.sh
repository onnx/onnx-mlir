git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout f9031f00f2c90bc0af274b45ec3e169b5250a688 && cd ..
