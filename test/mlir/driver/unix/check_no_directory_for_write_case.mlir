// RUN: (onnx-mlir ../../../../docs/doc_example/add.onnx -o NOOUTDIR/add.so || true) 2>&1 | FileCheck %s

// CHECK:      No such file or directory(2) for NOOUTDIR/add.so.unoptimized.bc at
