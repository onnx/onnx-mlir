// RUN: onnx-mlir --tag="encoder_model" --printIR %s | FileCheck %s

// CHECK: module attributes {{{.*}}"onnx-mlir.symbol-postfix" = "encoder_model"}
module {
}
