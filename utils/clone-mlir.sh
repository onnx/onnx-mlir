git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout f142f8afe21bceb00fb495468aa0b5043e98c419 && cd ..
