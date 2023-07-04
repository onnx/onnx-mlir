git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout bbeda83090adcb3609f9c1331b2345e7fa547f90 && cd ..
