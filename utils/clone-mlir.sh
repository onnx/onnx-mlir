git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 113f01aa82d055410f22a9d03b3468fa68600589 && cd ..
