// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

func.func private @none() {
  %0 = "krnl.noValue"() : () -> none
  return
}

// // -----

func.func private @none() {
  /// %none = "krnl.noValue"() : () -> none
  /// %f0 = arith.constant 0.0 : f32
  ///"krnl.call"(%none, %f0) {funcName = "func2", numOfOutput = 1 : si64} : (none, f32) -> ()
  %none = "krnl.noValue"() : () -> none
  "krnl.call"(%none) {funcName = "func2", numOfOutput = 1 : si64} : (none) -> ()
  return
}
