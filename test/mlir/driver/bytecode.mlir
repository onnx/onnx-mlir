// RUN: onnx-mlir --tag="encoder_model" --printBytecode %s | onnx-mlir-opt | FileCheck %s

// CHECK: module attributes {{{.*}}"onnx-mlir.symbol-postfix" = "encoder_model"}
module {
}
