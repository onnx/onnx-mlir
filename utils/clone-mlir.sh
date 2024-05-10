git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout fda272652fd65e139ed162a9c7ce521133eb34a0 && cd ..
