// RUN: onnx-mlir --printIR %s | FileCheck %s

// CHECK: module attributes {llvm.data_layout = "e-{{.*}}"}
module {
}
