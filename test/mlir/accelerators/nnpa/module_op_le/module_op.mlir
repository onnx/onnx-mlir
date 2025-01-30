// RUN: onnx-mlir --march=z16 --accel=NNPA --printIR %s | FileCheck %s

// CHECK: module attributes {llvm.data_layout = "e-{{.*}}", "onnx-mlir.symbol-postfix" = "{{.*}}"}
module {
}
