# Export environment variables pointing to LLVM-Projects.
export LLVM_PROJ_SRC=$(pwd)/llvm-project/
export LLVM_PROJ_BUILD=$(pwd)/llvm-project/build

if [ ! -d onnx-mlir/build ]; then
    mkdir onnx-mlir/build
fi
cd onnx-mlir/build
cmake ..
cmake --build . --target onnx-mlir

# Run FileCheck tests:
export LIT_OPTS=-v
cmake --build . --target check-mlir-lit