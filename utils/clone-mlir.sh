git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout f66cd9e9556a53142a26a5c21a72e21f1579217c && cd ..
