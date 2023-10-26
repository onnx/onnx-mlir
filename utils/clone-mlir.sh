git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout d13da154a7c7eff77df8686b2de1cfdfa7cc7029 && cd ..
