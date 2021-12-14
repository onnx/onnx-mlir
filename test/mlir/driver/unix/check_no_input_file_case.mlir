// RUN: (onnx-mlir noinput.onnx || exit 0) 2>&1 | FileCheck %s

// REQUIRES: system-linux
// CHECK:     Unable to open or access noinput.onnx
