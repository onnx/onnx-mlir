// RUN: onnx-mlir-opt --convert-krnl-to-affine %s | FileCheck %s

// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0)>
// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 6)>
// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 5)>

func @test_kernel_substitution() {
// CHECK-LABEL:   test_kernel_substitution
// CHECK:           affine.for [[VAR_arg0:%.+]] = 0 to 32 step 8 {
// CHECK:             affine.for [[VAR_arg1:%.+]] = 0 to 18 step 6 {
// CHECK:               [[VAR_0:%.+]] = alloc() : memref<10xf32>
// CHECK:               affine.for [[VAR_arg2:%.+]] = 0 to 20 step 5 {
// CHECK:                 [[VAR_1:%.+]] = alloca() : memref<10x10xf32>
// CHECK:                 [[VAR_2:%.+]] = alloca() : memref<10x8xf32>
// CHECK:                 affine.for [[VAR_arg3:%.+]] = #map0([[VAR_arg0]]) to #map1([[VAR_arg0]]) step 4 {
// CHECK:                   affine.for [[VAR_arg4:%.+]] = #map0([[VAR_arg1]]) to #map2([[VAR_arg1]]) step 3 {
// CHECK:                     affine.for [[VAR_arg5:%.+]] = #map0([[VAR_arg2]]) to #map3([[VAR_arg2]]) step 2 {
// CHECK:                       krnl.specialized_kernel() :
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:                 dealloc [[VAR_1]] : memref<10x10xf32>
// CHECK:                 dealloc [[VAR_2]] : memref<10x8xf32>
// CHECK:               }
// CHECK:               dealloc [[VAR_0]] : memref<10xf32>
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
    %alloc = alloc() : memref<10 x f32>
    krnl.iterate(%kb) with () {
      %Abuff = alloca(): memref<10x10xf32>
      %Bbuff = alloca(): memref<10x8xf32>
      krnl.iterate(%ilb, %jlb, %klb) with () {
        krnl.specialized_kernel(%ill, %jll, %kll) : !krnl.loop, !krnl.loop,!krnl.loop
      }
      dealloc %Abuff : memref<10x10xf32>
      dealloc %Bbuff : memref<10x8xf32>
    }
    dealloc %alloc : memref<10 x f32>
  }
  return
}

