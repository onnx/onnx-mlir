// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --disable-compiler-stick-unstick --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @should_lower_to_zlow_1d(%arg0: tensor<7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.Stick"(%arg0) {layout = "1D"} : (tensor<7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_1d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7xf32>) -> memref<7xf16, [[MAP_0_]]> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<7xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "1D"} : (memref<7xf32>, memref<7xf16, [[MAP_0_]]>) -> ()
// CHECK:           return [[RES_]] : memref<7xf16, [[MAP_0_]]>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_2d(%arg0: tensor<5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.Stick"(%arg0) {layout = "2D"} : (tensor<5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x7xf32>) -> memref<5x7xf16, [[MAP_0_]]> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5x7xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "2D"} : (memref<5x7xf32>, memref<5x7xf16, [[MAP_0_]]>) -> ()
// CHECK:           return [[RES_]] : memref<5x7xf16, [[MAP_0_]]>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_2ds(%arg0: tensor<5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.Stick"(%arg0) {layout = "2DS"} : (tensor<5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1 floordiv 64, 0, 0, 31, d1 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_2ds
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x7xf32>) -> memref<5x7xf16, [[MAP_0_]]> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5x7xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "2DS"} : (memref<5x7xf32>, memref<5x7xf16, [[MAP_0_]]>) -> ()
// CHECK:           return [[RES_]] : memref<5x7xf16, [[MAP_0_]]>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_3d(%arg0: tensor<3x5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.Stick"(%arg0) {layout = "3D"} : (tensor<3x5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (0, d2 floordiv 64, d0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_3d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5x7xf32>) -> memref<3x5x7xf16, [[MAP_0_]]> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x5x7xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "3D"} : (memref<3x5x7xf32>, memref<3x5x7xf16, [[MAP_0_]]>) -> ()
// CHECK:           return [[RES_]] : memref<3x5x7xf16, [[MAP_0_]]>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_3ds(%arg0: tensor<3x5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<3x5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_3ds
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5x7xf32>) -> memref<3x5x7xf16, [[MAP_0_]]> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x5x7xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "3DS"} : (memref<3x5x7xf32>, memref<3x5x7xf16, [[MAP_0_]]>) -> ()
// CHECK:           return [[RES_]] : memref<3x5x7xf16, [[MAP_0_]]>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_4d(%arg0: tensor<1x3x5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.Stick"(%arg0) {layout = "4D"} : (tensor<1x3x5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_4d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5x7xf32>) -> memref<1x3x5x7xf16, [[MAP_0_]]> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5x7xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "4D"} : (memref<1x3x5x7xf32>, memref<1x3x5x7xf16, [[MAP_0_]]>) -> ()
// CHECK:           return [[RES_]] : memref<1x3x5x7xf16, [[MAP_0_]]>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_nhwc(%arg0: tensor<1x3x5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x3x5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_nhwc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5x7xf32>) -> memref<1x5x7x3xf16, [[MAP_0_]]> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x5x7x3xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "NCHW"} : (memref<1x3x5x7xf32>, memref<1x5x7x3xf16, [[MAP_0_]]>) -> ()
// CHECK:           return [[RES_]] : memref<1x5x7x3xf16, [[MAP_0_]]>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_hwck(%arg0: tensor<1x3x5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.Stick"(%arg0) {layout = "HWCK"} : (tensor<1x3x5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3 floordiv 64, d0, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_hwck
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5x7xf32>) -> memref<1x3x5x7xf16, [[MAP_0_]]> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5x7xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "HWCK"} : (memref<1x3x5x7xf32>, memref<1x3x5x7xf16, [[MAP_0_]]>) -> ()
// CHECK:           return [[RES_]] : memref<1x3x5x7xf16, [[MAP_0_]]>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_fico(%arg0: tensor<1x5x7xf32>, %arg1: tensor<1x5x7xf32>, %arg2: tensor<1x5x7xf32>, %arg3: tensor<1x5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.StickForLSTM"(%arg0, %arg1, %arg2, %arg3) : (tensor<1x5x7xf32>, tensor<1x5x7xf32>, tensor<1x5x7xf32>, tensor<1x5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 7) * 57) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 7) * 57) mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_fico
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x5x7xf32>, [[PARAM_1_:%.+]]: memref<1x5x7xf32>, [[PARAM_2_:%.+]]: memref<1x5x7xf32>, [[PARAM_3_:%.+]]: memref<1x5x7xf32>) -> memref<1x5x28xf16, [[MAP_0_]]> {
 // CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x5x28xf16, [[MAP_0_]]>
 // CHECK:           "zlow.stickForLSTM"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[RES_]]) {prev_layer = "none"} : (memref<1x5x7xf32>, memref<1x5x7xf32>, memref<1x5x7xf32>, memref<1x5x7xf32>, memref<1x5x28xf16, [[MAP_0_]]>) -> ()
 // CHECK:           return [[RES_]] : memref<1x5x28xf16, [[MAP_0_]]>
 // CHECK:         }
}

// -----

func.func @should_lower_to_zlow_fico_bidir(%arg0: tensor<2x5x7xf32>, %arg1: tensor<2x5x7xf32>, %arg2: tensor<2x5x7xf32>, %arg3: tensor<2x5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.StickForLSTM"(%arg0, %arg1, %arg2, %arg3) : (tensor<2x5x7xf32>, tensor<2x5x7xf32>, tensor<2x5x7xf32>, tensor<2x5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 7) * 57) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 7) * 57) mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_fico_bidir
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x5x7xf32>, [[PARAM_1_:%.+]]: memref<2x5x7xf32>, [[PARAM_2_:%.+]]: memref<2x5x7xf32>, [[PARAM_3_:%.+]]: memref<2x5x7xf32>) -> memref<2x5x28xf16, [[MAP_0_]]> {
 // CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x5x28xf16, [[MAP_0_]]>
 // CHECK:           "zlow.stickForLSTM"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[RES_]]) {prev_layer = "none"} : (memref<2x5x7xf32>, memref<2x5x7xf32>, memref<2x5x7xf32>, memref<2x5x7xf32>, memref<2x5x28xf16, [[MAP_0_]]>) -> ()
 // CHECK:           return [[RES_]] : memref<2x5x28xf16, [[MAP_0_]]>
 // CHECK:         }
}

// -----

func.func @should_lower_to_zlow_zrh(%arg0: tensor<2x5x7xf32>, %arg1: tensor<2x5x7xf32>, %arg2: tensor<2x5x7xf32>) -> tensor<*xf16> {
  %0 = "zhigh.StickForGRU"(%arg0, %arg1, %arg2) : (tensor<2x5x7xf32>, tensor<2x5x7xf32>, tensor<2x5x7xf32>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 7) * 57) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 7) * 57) mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_zrh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x5x7xf32>, [[PARAM_1_:%.+]]: memref<2x5x7xf32>, [[PARAM_2_:%.+]]: memref<2x5x7xf32>) -> memref<2x5x21xf16, [[MAP_0_]]> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x5x21xf16, [[MAP_0_]]>
// CHECK:           "zlow.stickForGRU"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[RES_]]) {prev_layer = "none"} : (memref<2x5x7xf32>, memref<2x5x7xf32>, memref<2x5x7xf32>, memref<2x5x21xf16, [[MAP_0_]]>) -> ()
// CHECK:           return [[RES_]] : memref<2x5x21xf16, [[MAP_0_]]>
// CHECK:         }
}

