git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout e864ac694540342d5e59f59c525c5082f2594fb8 && cd ..
