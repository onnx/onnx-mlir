# Export environment variables pointing to LLVM-Projects.
export MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir

mkdir onnx-mlir/build && cd onnx-mlir/build
if [[ -n "$pythonLocation" ]]; then
  cmake -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DPython3_ROOT_DIR=$pythonLocation ..
else
  cmake -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/c++ ..
fi
cmake --build .

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
