// RUN: onnx-mlir --EmitLLVMIR --printIR %s | FileCheck %s

// Test that llvm.alloca is hoisted out of the loop nest.

// CHECK-LABEL: test_buffer_loop_hoisting
// CHECK-NOT: llvm.br
// CHECK-NOT: llvm.cond_br
// CHECK: llvm.alloca
// CHECK-NEXT: llvm.br
// CHECK: llvm.cond_br

func.func @test_buffer_loop_hoisting() {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  scf.for %arg0 = %c0 to %c32 step %c8 {
    %c18 = arith.constant 18 : index
    %c6 = arith.constant 6 : index
    scf.for %arg1 = %c0 to %c18 step %c6 {
      %c20 = arith.constant 20 : index
      %c5 = arith.constant 5 : index
      scf.for %arg2 = %c0 to %c20 step %c5 {
        %0 = memref.alloca() : memref<10x10xf32>
        %1 = memref.dim %0, %c0 : memref<10x10xf32>
        memref.dealloc %0 : memref<10x10xf32>
      }
    }
  }
  return
}
