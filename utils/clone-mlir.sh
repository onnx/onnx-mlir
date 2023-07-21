git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout ec89cb9a81529fd41fb37b8e62203a2e9f23bd54 && cd ..
