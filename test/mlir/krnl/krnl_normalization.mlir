// RUN: onnx-mlir-opt --normalize-memrefs %s -split-input-file | FileCheck %s

#map_tile = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 32, d3 floordiv 32, d2 mod 32, d3 mod 32)>

// CHECK-LABEL: test_krnl_memcpy_norm
// CHECK-SAME: -> memref<1x16x1x1x32x32xf32> {
func @test_krnl_memcpy_norm(%arg0: memref<1x256xf32>) -> memref<1x16x4x4xf32, #map_tile> {
  %0 = alloc() : memref<1x16x4x4xf32>
  %1 = alloc() : memref<1x16x4x4xf32, #map_tile>
  // CHECK: [[ALLOC:%.+]] = alloc() : memref<1x16x1x1x32x32xf32>
  %c256 = constant 256 : i64
  "krnl.memcpy"(%0, %arg0, %c256) : (memref<1x16x4x4xf32>, memref<1x256xf32>, i64) -> ()
  %ii, %ij, %ik, %il = krnl.define_loops 4
  krnl.iterate(%ii, %ij, %ik, %il)
    with (%ii -> %i = 0 to 1, %ij -> %j = 0 to 16, %ik -> %k = 0 to 4, %il -> %l = 0 to 4) {
    %2 = affine.load %0[%i, %j, %k, %l] : memref<1x16x4x4xf32>
    affine.store %2, %1[%i, %j, %k, %l] : memref<1x16x4x4xf32, #map_tile>
    // CHECK: affine.store
    // CHECK-SAME: : memref<1x16x1x1x32x32xf32>
  }
  dealloc %0 : memref<1x16x4x4xf32>
  return %1 :  memref<1x16x4x4xf32, #map_tile>
  // CHECK: return [[ALLOC]] : memref<1x16x1x1x32x32xf32>
}
