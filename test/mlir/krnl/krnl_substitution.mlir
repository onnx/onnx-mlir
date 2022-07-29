// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s | FileCheck %s

// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0)>
// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 6)>
// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 5)>

func.func @test_kernel_substitution() {
// CHECK-LABEL:   test_kernel_substitution
// CHECK:           affine.for [[I_L2_TILE:%.+]] = 0 to 32 step 8 {
// CHECK:             affine.for [[J_L2_TILE:%.+]] = 0 to 18 step 6 {
// CHECK:               [[L2_CACHE_TILE:%.+]] = memref.alloc() : memref<10xf32>
// CHECK:               affine.for [[K_L2_TILE:%.+]] = 0 to 20 step 5 {
// CHECK:                 [[L1_CACHE_TILE_A:%.+]] = memref.alloca() : memref<10x10xf32>
// CHECK:                 [[L1_CACHE_TILE_B:%.+]] = memref.alloca() : memref<10x8xf32>
// CHECK:                 affine.for [[I_L1_TILE:%.+]] = #map0([[I_L2_TILE]]) to #map1([[I_L2_TILE]]) step 4 {
// CHECK:                   affine.for [[J_L1_TILE:%.+]] = #map0([[J_L2_TILE]]) to #map2([[J_L2_TILE]]) step 3 {
// CHECK:                     affine.for [[K_L1_TILE:%.+]] = #map0([[K_L2_TILE]]) to #map3([[K_L2_TILE]]) step 2 {
// CHECK:                       krnl.specialized_kernel() :
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:                 memref.dealloc [[L1_CACHE_TILE_A]] : memref<10x10xf32>
// CHECK:                 memref.dealloc [[L1_CACHE_TILE_B]] : memref<10x8xf32>
// CHECK:               }
// CHECK:               memref.dealloc [[L2_CACHE_TILE]] : memref<10xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
  %ii, %ij, %ik = krnl.define_loops 3

  // 2-level Tile i loop:
  %ib, %il = krnl.block %ii 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %ilb, %ill = krnl.block %il 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)

  // 2-level Tile j loop:
  %jb, %jl = krnl.block %ij 6 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %jlb, %jll = krnl.block %jl 3 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)

  // 2-level Tile k loop:
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
}

