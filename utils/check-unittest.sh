#!/usr/bin/env bash
# Build and run unit tests
cd onnx-mlir/build
cmake --build . --target check-unittest
