// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

func.func private @test_krnl_call_with_novalue() {
  %none = "krnl.noValue"() : () -> none
  "krnl.call"(%none) {funcName = "func1", numOfOutput = 0 : si64} : (none) -> ()
  return

// mlir2FileCheck.py
// CHECK-LABEL:         llvm.func @test_krnl_call_with_novalue() attributes {llvm.emit_c_interface, sym_visibility = "private"} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.call @func1([[VAR_1_]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

}
