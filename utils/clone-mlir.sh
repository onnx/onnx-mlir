git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 43574226b712f94de17f07392a64282ac8ddab23 && cd ..
