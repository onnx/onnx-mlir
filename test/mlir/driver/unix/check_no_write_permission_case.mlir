// RUN: (rm -rf OUTDIR || mkdir -p OUTDIR || chmod -w OUTDIR || true 2>&1 > /dev/null;onnx-mlir ../../../../docs/doc_example/add.onnx -o OUTDIR/add.so || true; rm -rf OUTDIR || true 2>&1 > /dev/null) 2>&1 | FileCheck %s

// REQUIRES: system-linux
// CHECK:     No such file or directory(2) for OUTDIR/add.so.unoptimized.bc at
