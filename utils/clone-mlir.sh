git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout f580901d5d30e37755212f1c09e5b587587fbfeb && cd ..
