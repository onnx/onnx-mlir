#!/bin/bash

pwd
ls -al

# Install prereqs
pip install -e onnx-mlir/third_party/onnx

cd onnx-mlir/build

# Run end-to-end tests:
cmake --build . --target check-onnx-backend

# Run unit tests:
cmake --build .
cmake --build . --target test
