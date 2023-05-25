git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 095e6ac9fd92d03dcb1e19b60cb06a8140aae69 && cd ..
