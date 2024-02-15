// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s


func.func private @test_krnl_call_with_novalue(%arg0: memref<10x10xf32>) {
  %none = "krnl.noValue"() : () -> none
  "krnl.call"(%arg0, %none) {funcName = "func_with_value", numOfOutput = 1 : si64} : (memref<10x10xf32>, none) -> ()
  return
}
