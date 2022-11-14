// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func.func @simple_imperfectly_nested() {
  %ii = krnl.define_loops 1
  %ib, %il = krnl.block %ii 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%ib) with (%ii -> %i = 0 to 10) {
    %alloc = memref.alloc() : memref<10 x f32>
    krnl.iterate(%il) with () {
      %il_idx = krnl.get_induction_var_value(%il) : (!krnl.loop) -> (index)
      %v0 = krnl.load %alloc[%il_idx] : memref<10xf32>
      %foo = arith.addf %v0, %v0 : f32
    }
    memref.dealloc %alloc : memref<10 x f32>
  }
  return

// CHECK-LABEL: func @simple_imperfectly_nested
// CHECK-SAME:     () attributes {llvm.emit_c_interface} {
// CHECK:           affine.for [[I_BLOCK:%.+]] = 0 to 10 step 2 {
// CHECK:             [[ALLOC:%.+]] = memref.alloc() : memref<10xf32>
// CHECK:             affine.for [[I_LOCAL:%.+]] = #map([[I_BLOCK]]) to #map1([[I_BLOCK]]) {
// CHECK:               [[LOAD_VAL:%.+]] = affine.load [[ALLOC]]{{.}}[[I_LOCAL]]{{.}} : memref<10xf32>
// CHECK:               [[SUM:%.+]] = arith.addf [[LOAD_VAL]], [[LOAD_VAL]] : f32
// CHECK:             }
// CHECK:             memref.dealloc [[ALLOC]] : memref<10xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }
}

// -----

func.func @test_2d_tiling_imperfectly_nested() {
  %ii, %ij = krnl.define_loops 2
  %ib, %il = krnl.block %ii 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %ij 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%ib, %il, %jb, %jl) [0, 2, 1, 3] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 10, %ij -> %j = 0 to 20) {
    %alloc = memref.alloc() : memref<10 x f32>
    krnl.iterate(%il, %jl) with () {
      %il_idx, %jl_idx = krnl.get_induction_var_value(%il, %jl) : (!krnl.loop, !krnl.loop) -> (index, index)
      %foo = arith.addi %il_idx, %jl_idx : index
    }
    memref.dealloc %alloc : memref<10 x f32>
  }
  return

  // CHECK:       #map = affine_map<(d0) -> (d0)>
  // CHECK:       #map1 = affine_map<(d0) -> (d0 + 5)>
  // CHECK:       #map2 = affine_map<(d0) -> (d0 + 4)>
  // CHECK-LABEL:       func @test_2d_tiling_imperfectly_nested
  // CHECK-SAME:     () attributes {llvm.emit_c_interface} {
  // CHECK:           affine.for [[IB:%.+]] = 0 to 10 step 5 {
  // CHECK:             affine.for [[JB:%.+]] = 0 to 20 step 4 {
  // CHECK:               [[ALLOC:%.+]] = memref.alloc() : memref<10xf32>
  // CHECK:               affine.for [[IL:%.+]] = #map([[IB]]) to #map1([[IB]]) {
  // CHECK:                 affine.for [[JL:%.+]] = #map([[JB]]) to #map2([[JB]]) {
  // CHECK:                   [[FOO:%.+]] = arith.addi [[IL]], [[JL]] : index
  // CHECK:                 }
  // CHECK:               }
  // CHECK:               memref.dealloc [[ALLOC]] : memref<10xf32>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
}
