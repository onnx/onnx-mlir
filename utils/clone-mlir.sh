git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 853e0aa424e40b80d0bda1dd8a3471a361048e4b && cd ..
