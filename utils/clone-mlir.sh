git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 60a7d33106d3cd645d3100a8a935a1e3837f885d && cd ..
