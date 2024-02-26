git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout e899641df2391179e8ec29ca14c53b09ae7ce85c && cd ..
