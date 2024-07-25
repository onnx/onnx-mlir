git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 0913547d0e3939cc420e88ecd037240f33736820 && cd ..
