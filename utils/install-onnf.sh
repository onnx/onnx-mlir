# Export environment variables pointing to LLVM-Projects.
export LLVM_PROJ_SRC=$(pwd)/llvm-project/
export LLVM_PROJ_BUILD=$(pwd)/llvm-project/build

mkdir ONNF/build && cd ONNF/build
cmake ..
cmake --build . --target onnf

# Run FileCheck tests:
export LIT_OPTS=-v
cmake --build . --target check-mlir-lit