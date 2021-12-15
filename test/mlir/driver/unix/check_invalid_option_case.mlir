// RUN: (onnx-mlir --afo ../../../../docs/doc_example/add.onnx || exit 0) 2>&1 | FileCheck %s

// CHECK:      onnx-mlir: Unknown command line argument '--afo'.  Try:
// CHECK-NEXT: onnx-mlir: Did you mean '-o'?
