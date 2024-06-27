git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout c07be08df5731dac0b36e029a0dd03ccb099deea && cd ..
