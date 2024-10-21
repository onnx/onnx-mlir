git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 00128a20eec27246719d73ba427bf821883b00b4 && cd ..
