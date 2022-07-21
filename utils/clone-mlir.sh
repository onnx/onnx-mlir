git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout f2b94bd7eaa83d853dc7568fac87b1f8bf4ddec6 && cd ..
