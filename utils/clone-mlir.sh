git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 20ed5b1f45871612570d3bd447121ac43e083c6a && cd ..
