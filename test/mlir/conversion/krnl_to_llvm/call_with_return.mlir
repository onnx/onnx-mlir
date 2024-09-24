// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

func.func private @test_krnl_call_with_return(%arg0: memref<2x3xi32>) -> i32 {
  %1 = "krnl.call"() {funcName = "get_omp_num_thread", numOfOutput = 0 : si64} : () -> (i32)
  func.return %1: i32
// CHECK:         llvm.func @get_omp_num_thread() -> i32
// CHECK:         llvm.func @test_krnl_call_with_return
// CHECK:           [[VAR_0_:%.+]] = llvm.call @get_omp_num_thread() : () -> i32
// CHECK:           llvm.return [[VAR_0_]] : i32
}
