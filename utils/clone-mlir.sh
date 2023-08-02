git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 438fc2c83b73e66f6dbae4f34e9a19f41302f825 && cd ..
