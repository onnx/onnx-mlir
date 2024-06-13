git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout c012e487b7246239c31bd378ab074fb110631186 && cd ..
