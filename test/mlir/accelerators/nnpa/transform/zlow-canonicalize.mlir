// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --canonicalize %s -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @remove_unused_stick_op(%arg0: memref<5x10xf32>) -> memref<10xf32> {
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf16, #map>
  "zlow.stick"(%arg0, %0) {layout = "2D"} : (memref<5x10xf32>, memref<5x10xf16, #map>) -> ()
  %1 = memref.alloc() {alignment = 4096 : i64} : memref<10xf32>
  return %1: memref<10xf32>

// CHECK-LABEL:  func.func @remove_unused_stick_op
// CHECK-NEXT:   memref.alloc
// CHECK-NEXT:   return 
// CHECK-NOT:    zlow.stick 
}

// -----

#map = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @donot_remove_stick_op(%arg0: memref<5x10xf32>) -> memref<5x10xf16, #map> {
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf16, #map>
  "zlow.stick"(%arg0, %0) {layout = "2D"} : (memref<5x10xf32>, memref<5x10xf16, #map>) -> ()
  return %0: memref<5x10xf16, #map>

// CHECK-LABEL:  func.func @donot_remove_stick_op
// CHECK-NEXT:   memref.alloc
// CHECK-NEXT:   zlow.stick 
// CHECK-NEXT:   return 
}
// -----

#map = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @remove_unused_unstick_op(%arg0: memref<5x10xf16, #map>) -> memref<10xf32> {
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf32>
  "zlow.unstick"(%arg0, %0) {layout = "2D"} : (memref<5x10xf16, #map>, memref<5x10xf32>) -> ()
  %1 = memref.alloc() {alignment = 4096 : i64} : memref<10xf32>
  return %1: memref<10xf32>

// CHECK-LABEL:  func.func @remove_unused_unstick_op
// CHECK-NEXT:   memref.alloc
// CHECK-NEXT:   return 
// CHECK-NOT:    zlow.unstick 
}

// -----

#map = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @donot_remove_unstick_op(%arg0: memref<5x10xf16, #map>) -> memref<5x10xf32> {
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<5x10xf32>
  "zlow.unstick"(%arg0, %0) {layout = "2D"} : (memref<5x10xf16, #map>, memref<5x10xf32>) -> ()
  return %0: memref<5x10xf32>

// CHECK-LABEL:  func.func @donot_remove_unstick_op
// CHECK-NEXT:   memref.alloc
// CHECK-NEXT:   zlow.unstick 
// CHECK-NEXT:   return 
}

