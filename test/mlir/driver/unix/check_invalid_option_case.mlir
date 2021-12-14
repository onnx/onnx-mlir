// RUN: (onnx-mlir --afo ../../../../docs/doc_example/add.onnx || exit 0) 2>&1 | FileCheck %s

// REQUIRES: system-linux
// CHECK:      onnx-mlir: Unknown command line argument '--afo'.  Try: '/home/negishi/src/dlc.git/onnx-mlir/build/Debug/bin/onnx-mlir --help'
// CHECK-NEXT: onnx-mlir: Did you mean '-o'?
