// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// CHECK: module attributes {llvm.data_layout = "E"}
module {
}

