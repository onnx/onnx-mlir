git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout ee40aefab5759eae6feacab6612bc628ae5d2993 && cd ..
