git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 74fb770de9399d7258a8eda974c93610cfde698e && cd ..
