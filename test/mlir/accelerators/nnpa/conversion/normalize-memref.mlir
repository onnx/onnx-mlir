// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --normalize-memrefs %s -split-input-file | FileCheck %s

// -----

#map0 = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>

func.func @test_zlow_relu_norm(%arg0: memref<129x65xf32>) -> memref<129x65xf32> {
  %0 = memref.alloc() : memref<129x65xf32>
  %1 = memref.alloc() {alignment = 4096 : i64} : memref<129x65xf16, #map0>
  %2 = memref.alloc() {alignment = 4096 : i64} : memref<129x65xf16, #map0>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.stick"(%arg0, %2) : (memref<129x65xf32>, memref<129x65xf16, #map0>) -> ()
  "zlow.relu"(%2, %shape, %1) { layout = "2D" } : (memref<129x65xf16, #map0>, memref<2xi64>, memref<129x65xf16, #map0>) -> ()
  "zlow.unstick"(%1, %0) : (memref<129x65xf16, #map0>, memref<129x65xf32>) -> ()
  memref.dealloc %2 : memref<129x65xf16, #map0>
  memref.dealloc %1 : memref<129x65xf16, #map0>
  return %0 : memref<129x65xf32>

  // CHECK-LABEL: test_zlow_relu_norm
  // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<129x65xf32>
  // CHECK: [[ALLOC2:%.+]] = memref.alloc() {alignment = 4096 : i64} : memref<1x2x1x5x32x64xf16>
  // CHECK: [[ALLOC3:%.+]] = memref.alloc() {alignment = 4096 : i64} : memref<1x2x1x5x32x64xf16>
  // CHECK: [[SHAPE:%.+]] = memref.alloc() : memref<2xi64>
  // CHECK: "zlow.stick"(%arg0, [[ALLOC3]]) : (memref<129x65xf32>, memref<1x2x1x5x32x64xf16>) -> ()
  // CHECK: "zlow.relu"([[ALLOC3]], [[SHAPE]], [[ALLOC2]]) {layout = "2D"} : (memref<1x2x1x5x32x64xf16>, memref<2xi64>, memref<1x2x1x5x32x64xf16>) -> ()
  // CHECK: "zlow.unstick"([[ALLOC2]], [[ALLOC1]]) : (memref<1x2x1x5x32x64xf16>, memref<129x65xf32>) -> ()
  // CHECK: memref.dealloc [[ALLOC3]] : memref<1x2x1x5x32x64xf16>
  // CHECK: memref.dealloc [[ALLOC2]] : memref<1x2x1x5x32x64xf16>
  // CHECK: return [[ALLOC1]] : memref<129x65xf32>
}

// -----

#map_1d = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
#map_2d = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
#map_2ds = affine_map<(d0, d1) -> (d0, d1 floordiv 64, 0, 0, 31, d1 mod 64)>
#map_3d = affine_map<(d0, d1, d2) -> (0, d2 floordiv 64, d0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
#map_3ds = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
#map_4d = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
#map_nchw = affine_map<(d0, d1, d2, d3) -> (d0, d2 floordiv 64, d3, d1 floordiv 32, d1 mod 32, d2 mod 64)>

// CHECK-LABEL: test_stick_norm
func.func @test_stick_norm(%arg1d: memref<129xf32>, %arg2d: memref<129x65xf32>,
                      %arg2ds: memref<129x65xf32>, %arg3d: memref<129x65x129xf32>,
                      %arg3ds: memref<129x65x129xf32>, %arg4d: memref<129x65x129x65xf32>,
                      %argnchw: memref<129x65x129x65xf32>  ) -> () {
  // CHECK-NEXT: memref<1x3x1x1x32x64xf16>
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<129xf16, #map_1d>
  // CHECK-NEXT: memref<1x3x1x1x32x64xf16>
  "zlow.stick"(%arg1d, %0) : (memref<129xf32>, memref<129xf16, #map_1d>) -> ()

  // CHECK-NEXT: memref<1x2x1x5x32x64xf16>
  %1 = memref.alloc() {alignment = 4096 : i64} : memref<129x65xf16, #map_2d>
  // CHECK-NEXT: memref<1x2x1x5x32x64xf16>
  "zlow.stick"(%arg2d, %1) : (memref<129x65xf32>, memref<129x65xf16, #map_2d>) -> ()

  // CHECK-NEXT: memref<129x2x1x1x32x64xf16>
  %2 = memref.alloc() {alignment = 4096 : i64} : memref<129x65xf16, #map_2ds>
  // CHECK-NEXT: memref<129x2x1x1x32x64xf16>
  "zlow.stick"(%arg2ds, %2) : (memref<129x65xf32>, memref<129x65xf16, #map_2ds>) -> ()

  // CHECK-NEXT: memref<1x3x129x3x32x64xf16>
  %3 = memref.alloc() {alignment = 4096 : i64} : memref<129x65x129xf16, #map_3d>
  // CHECK-NEXT: memref<1x3x129x3x32x64xf16>
  "zlow.stick"(%arg3d, %3) : (memref<129x65x129xf32>, memref<129x65x129xf16, #map_3d>) -> ()

  // CHECK-NEXT: memref<129x3x1x3x32x64xf16>
  %4 = memref.alloc() {alignment = 4096 : i64} : memref<129x65x129xf16, #map_3ds>
  // CHECK-NEXT: memref<129x3x1x3x32x64xf16>
  "zlow.stick"(%arg3ds, %4) : (memref<129x65x129xf32>, memref<129x65x129xf16, #map_3ds>) -> ()

  // CHECK-NEXT: memref<129x2x65x5x32x64xf16>
  %5 = memref.alloc() {alignment = 4096 : i64} : memref<129x65x129x65xf16, #map_4d>
  // CHECK-NEXT: memref<129x2x65x5x32x64xf16>
  "zlow.stick"(%arg4d, %5) : (memref<129x65x129x65xf32>, memref<129x65x129x65xf16, #map_4d>) -> ()

  // CHECK-NEXT: memref<129x3x65x3x32x64xf16>
  %6 = memref.alloc() {alignment = 4096 : i64} : memref<129x65x129x65xf16, #map_nchw>
  // CHECK-NEXT: memref<129x3x65x3x32x64xf16>
  "zlow.stick"(%argnchw, %6) : (memref<129x65x129x65xf32>, memref<129x65x129x65xf16, #map_nchw>) -> ()
  return
}
