// RUN: onnx-mlir-opt --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func @simple_imperfectly_nested() {
  %ii = krnl.define_loops 1
  %ib, %il = krnl.block %ii 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%ib) with (%ii -> %i = 0 to 10) {
    %alloc = memref.alloc() : memref<10 x f32>
    krnl.iterate(%il) with () {
      %v0 = krnl.load %alloc[%i] : memref<10xf32>
      %foo = addf %v0, %v0 : f32
    }
    memref.dealloc %alloc : memref<10 x f32>
  }
  return

// CHECK-LABEL: func @simple_imperfectly_nested
// CHECK-SAME:     () {
// CHECK:           affine.for [[I_BLOCK:%.+]] = 0 to 10 step 2 {
// CHECK:             [[ALLOC:%.+]] = memref.alloc() : memref<10xf32>
// CHECK:             affine.for [[I_LOCAL:%.+]] = #map0([[I_BLOCK]]) to #map1([[I_BLOCK]]) {
// CHECK:               [[LOAD_VAL:%.+]] = affine.load [[ALLOC]]{{.}}[[I_LOCAL]]{{.}} : memref<10xf32>
// CHECK:               [[SUM:%.+]] = addf [[LOAD_VAL]], [[LOAD_VAL]] : f32
// CHECK:             }
// CHECK:             memref.dealloc [[ALLOC]] : memref<10xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }
}

// -----

func @test_2d_tiling_imperfectly_nested() {
  %ii, %ij = krnl.define_loops 2
  %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
    %alloc = memref.alloc() : memref<10 x f32>
    krnl.iterate(%il, %jl) with () {
      %foo = addi %i, %j : index
    }
    memref.dealloc %alloc : memref<10 x f32>
  }
  return

  // CHECK:       #map0 = affine_map<(d0) -> (d0)>
  // CHECK:       #map1 = affine_map<(d0) -> (d0 + 5)>
  // CHECK:       #map2 = affine_map<(d0) -> (d0 + 4)>
  // CHECK-LABEL:       func @test_2d_tiling_imperfectly_nested
  // CHECK-SAME:     () {
  // CHECK:           affine.for [[IB:%.+]] = 0 to 10 step 5 {
  // CHECK:             affine.for [[JB:%.+]] = 0 to 20 step 4 {
  // CHECK:               [[ALLOC:%.+]] = memref.alloc() : memref<10xf32>
  // CHECK:               affine.for [[IL:%.+]] = #map0([[IB]]) to #map1([[IB]]) {
  // CHECK:                 affine.for [[JL:%.+]] = #map0([[JB]]) to #map2([[JB]]) {
  // CHECK:                   [[FOO:%.+]] = addi [[IL]], [[JL]] : index
  // CHECK:                 }
  // CHECK:               }
  // CHECK:               memref.dealloc [[ALLOC]] : memref<10xf32>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
}