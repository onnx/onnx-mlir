git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 438e59182b0c2e44c263f5bacc1add0e514354f8 && cd ..
