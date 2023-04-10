// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s | FileCheck %s

// -----

func.func @test_kernel_substitution() {
  %ii, %ij, %ik = krnl.define_loops 3
  %ib, %il = krnl.block %ii 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %ilb, %ill = krnl.block %il 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jb, %jl = krnl.block %ij 6 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jlb, %jll = krnl.block %jl 3 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %kb, %kl = krnl.block %ik 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %klb, %kll = krnl.block %kl 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%ib, %ilb, %ill, %jb, %jlb, %jll, %kb, %klb, %kll) [0, 3, 6, 1, 4, 7, 2, 5, 8 ] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 32, %ij -> %j = 0 to 18, %ik -> %k = 0 to 20) {
    %alloc = memref.alloc() : memref<10 x f32>
    krnl.iterate(%kb) with () {
      %Abuff = memref.alloca(): memref<10x10xf32>
      %Bbuff = memref.alloca(): memref<10x8xf32>
      krnl.iterate(%ilb, %jlb, %klb) with () {
        krnl.specialized_kernel(%ill, %jll, %kll) : !krnl.loop, !krnl.loop,!krnl.loop
      }
      memref.dealloc %Abuff : memref<10x10xf32>
      memref.dealloc %Bbuff : memref<10x8xf32>
    }
    memref.dealloc %alloc : memref<10 x f32>
  }
  return

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 6)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 + 5)>
// CHECK-LABEL:  func.func @test_kernel_substitution
// CHECK-SAME:   () attributes {llvm.emit_c_interface} {
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 32 step 8 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 18 step 6 {
// CHECK:               [[RES_:%.+]] = memref.alloc() : memref<10xf32>
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 20 step 5 {
// CHECK-DAG:             [[RES_1_:%.+]] = memref.alloca() : memref<10x10xf32>
// CHECK-DAG:             [[RES_2_:%.+]] = memref.alloca() : memref<10x8xf32>
// CHECK:                 affine.for [[I_3_:%.+]] = [[MAP_0_]]([[I_0_]]) to [[MAP_2_]]([[I_0_]]) step 4 {
// CHECK:                   affine.for [[I_4_:%.+]] = [[MAP_0_]]([[I_1_]]) to [[MAP_3_]]([[I_1_]]) step 3 {
// CHECK:                     affine.for [[I_5_:%.+]] = [[MAP_0_]]([[I_2_]]) to [[MAP_4_]]([[I_2_]]) step 2 {
// CHECK:                       krnl.specialized_kernel() :
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:                 memref.dealloc [[RES_1_]] : memref<10x10xf32>
// CHECK:                 memref.dealloc [[RES_2_]] : memref<10x8xf32>
// CHECK:               }
// CHECK:               memref.dealloc [[RES_]] : memref<10xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

