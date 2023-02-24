git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 114ce273d86a091f61446c8778dbf102942c96cf && cd ..
