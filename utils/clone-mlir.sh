git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout b2cdf3cc4c08729d0ff582d55e40793a20bbcdcc && cd ..
