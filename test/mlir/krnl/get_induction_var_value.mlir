// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func.func @test_simple() {
  %ii = krnl.define_loops 1
  krnl.iterate(%ii) with (%ii -> %i = 0 to 10) {
    %idx = krnl.get_induction_var_value(%ii) : (!krnl.loop) -> (index)
    %foo = arith.addi %idx, %idx : index
  }
  return
  // CHECK-LABEL:       func @test_simple
  // CHECK-SAME:     () attributes {llvm.emit_c_interface} {
  // CHECK:           affine.for [[I:%.+]] = 0 to 10 {
  // CHECK:             [[FOO:%.+]] = arith.addi [[I]], [[I]] : index
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
}

// -----

func.func @test_2d_tiling_imperfectly_nested() {
  %ii, %ij = krnl.define_loops 2
  %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
    %ib_idx, %jb_idx = krnl.get_induction_var_value(%ib, %jb) : (!krnl.loop, !krnl.loop) -> (index, index)
    %bar = arith.addi %ib_idx, %jb_idx : index

    %alloc = memref.alloc() : memref<10 x f32>
    krnl.iterate(%il, %jl) with () {
      %il_idx, %jl_idx = krnl.get_induction_var_value(%il, %jl) : (!krnl.loop, !krnl.loop) -> (index, index)
      %foo = arith.addi %il_idx, %jl_idx : index
    }
    memref.dealloc %alloc : memref<10 x f32>
  }
  // CHECK-LABEL:       func @test_2d_tiling_imperfectly_nested
  // CHECK-SAME:     () attributes {llvm.emit_c_interface} {
  // CHECK:           affine.for [[IBLOCK:%.+]] = 0 to 10 step 5 {
  // CHECK:             affine.for [[JBLOCK:%.+]] = 0 to 20 step 4 {
  // CHECK:               [[BAR:%.+]] = arith.addi [[IBLOCK]], [[JBLOCK]] : index
  // CHECK:               [[TILE_BUFFER:%.+]] = memref.alloc() : memref<10xf32>
  // CHECK:               affine.for [[ILOCAL:%.+]] = #map([[IBLOCK]]) to #map1([[IBLOCK]]) {
  // CHECK:                 affine.for [[JLOCAL:%.+]] = #map([[JBLOCK]]) to #map2([[JBLOCK]]) {
  // CHECK:                   [[BAR:%.+]] = arith.addi [[ILOCAL]], [[JLOCAL]] : index
  // CHECK:                 }
  // CHECK:               }
  // CHECK:               memref.dealloc [[TILE_BUFFER]] : memref<10xf32>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
  return
}
