// RUN: onnx-mlir-opt --normalize-memrefs %s -split-input-file | FileCheck %s

#map_tile = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 32, d3 floordiv 32, d2 mod 32, d3 mod 32)>

// CHECK-LABEL: test_krnl_memcpy_norm
// CHECK-SAME: -> memref<1x16x1x1x32x32xf32> {
func @test_krnl_memcpy_norm(%arg0: memref<1x256xf32>) -> memref<1x16x4x4xf32, #map_tile> {
  %0 = alloc() : memref<1x16x4x4xf32, #map_tile>
  // CHECK: [[ALLOC:%.+]] = alloc() : memref<1x16x1x1x32x32xf32>
  %c16384 = constant 16384 : i64
  "krnl.memcpy"(%0, %arg0, %c16384) : (memref<1x16x4x4xf32, #map_tile>, memref<1x256xf32>, i64) -> ()
  // CHECK: "krnl.memcpy"
  // CHECK-SAME: : (memref<1x16x1x1x32x32xf32>, memref<1x256xf32>
  return %0 : memref<1x16x4x4xf32, #map_tile>
  // CHECK: return [[ALLOC]] : memref<1x16x1x1x32x32xf32>  
}
