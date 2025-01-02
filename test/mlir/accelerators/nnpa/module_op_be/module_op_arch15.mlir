// RUN: onnx-mlir --march=arch15 --maccel=NNPA --printIR %s | FileCheck %s

// CHECK: module attributes {llvm.data_layout = "E-{{.*}}", llvm.target_triple = "{{.*}}", "onnx-mlir.accels" = ["NNPA-0x10101"], "onnx-mlir.symbol-postfix" = "{{.*}}"}
module {
}
