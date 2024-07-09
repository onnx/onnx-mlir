git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 6461b921fd06b1c812f1172685b8b7edc0608af7 && cd ..
