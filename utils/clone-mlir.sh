# Check out a specific branch that is known to work with ONNX-MLIR

# Shallow fetch to avoid cloning the full history, shallow clone requires a relative recent git 2.49
git init llvm-project
cd llvm-project
git remote add origin https://github.com/xilinx/llvm-aie.git
git fetch --depth 1 origin 39149e4f236f7cb37d510e5d2445f86385ac3b6b
git checkout FETCH_HEAD
cd ..
