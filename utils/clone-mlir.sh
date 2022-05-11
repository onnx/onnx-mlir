git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 4a3460a7917f1cf514575759e29590b388131fc6 && cd ..
