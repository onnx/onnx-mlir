git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 4acc3ffbb0af5631bc7916aeff3570f448899647 && cd ..
