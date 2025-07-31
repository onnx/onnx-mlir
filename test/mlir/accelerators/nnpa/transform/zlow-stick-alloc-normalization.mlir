// RUN: onnx-mlir-opt --march=z17 --maccel=NNPA --zlow-stick-expansion="enable-stick-expansion=false enable-alloc-normalization=true" %s -split-input-file | FileCheck %s

// -----

// No alloc normalization possible for input arguments
#map = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
func.func @test_no_normalization(%arg0: memref<16x8x128xf32>) -> memref<16x8x128xf16, #map> {
  %alloc = memref.alloc() {alignment = 4096 : i64} : memref<16x8x128xf16, #map>
  "zlow.stick"(%arg0, %alloc) {layout = "3DS"} : (memref<16x8x128xf32>, memref<16x8x128xf16, #map>) -> ()
  return %alloc : memref<16x8x128xf16, #map>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func.func @test_no_normalization
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32>) -> memref<16x8x128xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {alignment = 4096 : i64} : memref<16x8x128xf16, #map>
// CHECK:         }
}
// -----

// Here the value being stickified is from an alloc memref in the model, so normalize to 4k.

#map = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
func.func @test_normalization(%arg0: memref<16x8x128xf32>) -> memref<16x8x128xf16, #map> {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<16x8x128xf32>
  affine.for %arg1 = 0 to 16 {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 128 {
        %0 = affine.load %arg0[%arg1, %arg2, %arg3] : memref<16x8x128xf32>
        %1 = math.sin %0 : f32
        affine.store %1, %alloc[%arg1, %arg2, %arg3] : memref<16x8x128xf32>
      }
    }
  }
  %alloc1 = memref.alloc() {alignment = 4096 : i64} : memref<16x8x128xf16, #map>
  "zlow.stick"(%alloc, %alloc1) {layout = "3DS", no_saturation = -1 : si64} : (memref<16x8x128xf32>, memref<16x8x128xf16, #map>) -> ()
  return %alloc1 : memref<16x8x128xf16, #map>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func.func @test_normalization
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8x128xf32>) -> memref<16x8x128xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {alignment = 4096 : i64} : memref<16x8x128xf32>
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {alignment = 4096 : i64} : memref<16x8x128xf16, #map>
}
