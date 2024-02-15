// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

func.func private @test_krnl_call_with_novalue() {
  %none = "krnl.noValue"() : () -> none
  "krnl.call"(%none) {funcName = "func1", numOfOutput = 0 : si64} : (none) -> ()
  return

// CHECK-LABEL: test_krnl_call_with_novalue
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       llvm.call @func1([[VAR_1_]]) : (!llvm.ptr) -> ()

}

// -----

func.func private @test_krnl_call_with_novalue_2(%arg0: memref<10x10xf32>) {
  %none = "krnl.noValue"() : () -> none
  "krnl.call"(%arg0, %none) {funcName = "func", numOfOutput = 1 : si64} : (memref<10x10xf32>, none) -> ()
  return

// CHECK-LABEL: test_krnl_call_with_novalue_2
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       llvm.call @func1(%arg0, [[VAR_1_]]) : (!llvm.ptr) -> ()

}
