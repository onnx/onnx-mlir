git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 7ac7d418ac2b16fd44789dcf48e2b5d73de3e715 && cd ..
