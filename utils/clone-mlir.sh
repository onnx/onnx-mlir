git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 6875424135312aeb26ab8e0358ba7f9e6e80e741 && cd ..
