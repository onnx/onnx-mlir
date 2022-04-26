# MLIR_DIR must be set with cmake option now
mkdir onnx-mlir/build && cd onnx-mlir/build
if [[ -z "$pythonLocation" ]]; then
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DMLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir \
        ..
else
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DPython3_ROOT_DIR=$pythonLocation \
        -DMLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir \
        ..
fi
cmake --build .

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
