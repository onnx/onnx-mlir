git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 0e779ad4998ef65907502101c5b82ede05ddfa4e && cd ..
