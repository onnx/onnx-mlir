#!/bin/bash

pwd
ls -al
cd onnx-mlir/build

# Run end-to-end tests:
cmake --build . --target check-onnx-backend

# Run unit tests:
cmake --build . --target test
