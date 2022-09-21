git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 458598ccc50c5118107f05d60f3d043772a91f26 && cd ..
