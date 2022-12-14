git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 9acc2f37bdfce08ca0c2faec03392db10d1bb7a9 && cd ..
