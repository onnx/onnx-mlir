git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout f8cb7987c64dcffb72414a40560055cb717dbf74 && cd ..
