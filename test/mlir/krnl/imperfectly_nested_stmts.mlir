// RUN: onnx-mlir-opt --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func @simple_block() {
  %ii = krnl.define_loops 1
  %ib, %il = krnl.block %ii 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%ib) with (%ii -> %i = 0 to 10) {
    %alloc = alloc() : memref<10 x f32>
    krnl.iterate(%il) with () {
      %v0 = krnl.load %alloc[%i] : memref<10xf32>
      %foo = addf %v0, %v0 : f32
    }
    dealloc %alloc : memref<10 x f32>
  }
  return

// CHECK-LABEL: func @simple_block
// CHECK-SAME:     () {
// CHECK:           affine.for [[I_BLOCK:%.+]] = 0 to 10 step 2 {
// CHECK:             [[ALLOC:%.+]] = alloc() : memref<10xf32>
// CHECK:             affine.for [[I_LOCAL:%.+]] = #map0([[I_BLOCK]]) to #map1([[I_BLOCK]]) {
// CHECK:               [[LOAD_VAL:%.+]] = affine.load [[ALLOC]]{{.}}[[I_LOCAL]]{{.}} : memref<10xf32>
// CHECK:               [[SUM:%.+]] = addf [[LOAD_VAL]], [[LOAD_VAL]] : f32
// CHECK:             }
// CHECK:             dealloc [[ALLOC]] : memref<10xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }
}