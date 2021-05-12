// RUN: onnx-mlir-opt --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func @test_simple() {
  %ii = krnl.define_loops 1
  krnl.iterate(%ii) with (%ii -> %i = 0 to 10) {
    %idx = krnl.get_induction_var_value(%ii) : (!krnl.loop) -> (index)
    %foo = addi %idx, %idx : index
  }
  return
  // CHECK-LABEL:       func @test_simple
  // CHECK-SAME:     () {
  // CHECK:           affine.for [[I:%.+]] = 0 to 10 {
  // CHECK:             [[FOO:%.+]] = addi [[I]], [[I]] : index
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
}

// -----

func @test_2d_tiling_imperfectly_nested() {
  %ii, %ij = krnl.define_loops 2
  %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
    %ib_idx, %jb_idx = krnl.get_induction_var_value(%ib, %jb) : (!krnl.loop, !krnl.loop) -> (index, index)
    %bar = addi %ib_idx, %jb_idx : index

    %alloc = memref.alloc() : memref<10 x f32>
    krnl.iterate(%il, %jl) with () {
      %foo = addi %i, %j : index
    }
    memref.dealloc %alloc : memref<10 x f32>
  }
  // CHECK-LABEL:       func @test_2d_tiling_imperfectly_nested
  // CHECK-SAME:     () {
  // CHECK:           affine.for [[IBLOCK:%.+]] = 0 to 10 step 5 {
  // CHECK:             affine.for [[JBLOCK:%.+]] = 0 to 20 step 4 {
  // CHECK:               [[BAR:%.+]] = addi [[IBLOCK]], [[JBLOCK]] : index
  // CHECK:               [[TILE_BUFFER:%.+]] = memref.alloc() : memref<10xf32>
  // CHECK:               affine.for [[ILOCAL:%.+]] = #map0([[IBLOCK]]) to #map1([[IBLOCK]]) {
  // CHECK:                 affine.for [[JLOCAL:%.+]] = #map0([[JBLOCK]]) to #map2([[JBLOCK]]) {
  // CHECK:                   [[BAR:%.+]] = addi [[ILOCAL]], [[JLOCAL]] : index
  // CHECK:                 }
  // CHECK:               }
  // CHECK:               memref.dealloc [[TILE_BUFFER]] : memref<10xf32>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
  return
}