git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout fc44a4fcd3c54be927c15ddd9211aca1501633e7 && cd ..
