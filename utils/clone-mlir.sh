git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 3abae1c88416858cf2e9a7ed9417bc52033933b4 && cd ..
