git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 6098d7d5f6533edb1b873107ddc1acde23b9235b && cd ..
