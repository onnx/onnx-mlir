#!/usr/bin/env bash
# Run backend and numerical tests in parallel
cd onnx-mlir/build
CTEST_PARALLEL_LEVEL=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu) \
cmake --build . --parallel --target check-onnx-backend check-onnx-backend-dynamic check-onnx-backend-input-verification check-onnx-numerical
