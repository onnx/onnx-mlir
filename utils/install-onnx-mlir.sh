# Export environment variables pointing to LLVM-Projects.
export LLVM_PROJ_SRC=$(pwd)/llvm-project/
export LLVM_PROJ_BUILD=$(pwd)/llvm-project/build

mkdir onnx-mlir/build && cd onnx-mlir/build
cmake ..
cmake --build .

# Run FileCheck tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
