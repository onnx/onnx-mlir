// RUN: onnx-mlir-opt --convert-krnl-to-affine="preprocessing-only=true"  --verify-each=0 %s -split-input-file | FileCheck %s
// Note that turning off verification is necessary - moving chunks of IRs around necessarily breaks dominance temporarily.

func @simple_imperfectly_nested() {
  %ii = krnl.define_loops 1
  %ib, %il = krnl.block %ii 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%ib) with (%ii -> %i = 0 to 10) {
    %alloc = memref.alloc() : memref<10 x f32>
    krnl.iterate(%il) with () {
      %il_idx = krnl.get_induction_var_value(%il) : (!krnl.loop) -> (index)
      %v0 = krnl.load %alloc[%il_idx] : memref<10xf32>
      %foo = addf %v0, %v0 : f32
    }
    memref.dealloc %alloc : memref<10 x f32>
  }
  return

  // CHECK-LABEL:  func @simple_imperfectly_nested
  // CHECK-SAME:     () {
  // CHECK:           [[VAR_0:%.+]] = krnl.define_loops 1
  // CHECK:           [[VAR_loop_block:%.+]], [[VAR_loop_local:%.+]] = krnl.block [[VAR_0]] 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           krnl.iterate([[VAR_loop_block]]) with ([[VAR_0]] -> [[VAR_arg0:%.+]] = 0 to 10) {
  // CHECK:             krnl.iterate([[VAR_loop_local]]) with () {
  // CHECK:             }
  // CHECK:           }
  // CHECK:           krnl.movable  {
  // CHECK:             [[VAR_1:%.+]] = krnl.get_induction_var_value([[VAR_loop_local]]) : (!krnl.loop) -> index
  // CHECK:             [[VAR_2:%.+]] = krnl.load [[VAR_1]]{{.}}[[VAR_1]]{{.}} : memref<10xf32>
  // CHECK:             [[VAR_3:%.+]] = addf [[VAR_2]], [[VAR_2]] : f32
  // CHECK:           }
  // CHECK:           krnl.movable  {
  // CHECK:             [[VAR_1:%.+]] = memref.alloc() : memref<10xf32>
  // CHECK:           }
  // CHECK:           krnl.movable  {
  // CHECK:             memref.dealloc [[VAR_1]] : memref<10xf32>
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
}