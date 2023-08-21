// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test krnl lowering for IsInf.
//===----------------------------------------------------------------------===//

