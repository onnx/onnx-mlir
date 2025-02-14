// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --disable-compiler-stick-unstick --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @should_lower_to_zlow(%arg0: tensor<1x3x5x7xf32>) -> tensor<*xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x3x5x7xf32>) -> tensor<*xf16>
  %1 = "zhigh.Unstick"(%0) : (tensor<*xf16>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5x7xf32>) -> memref<1x3x5x7xf32> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x5x7x3xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "NCHW"} : (memref<1x3x5x7xf32>, memref<1x5x7x3xf16, [[MAP_0_]]>) -> ()
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5x7xf32>
// CHECK:           "zlow.unstick"([[RES_]], [[RES_1_]]) {layout = "NCHW"} : (memref<1x5x7x3xf16, [[MAP_0_]]>, memref<1x3x5x7xf32>) -> ()
// CHECK:           return [[RES_1_]] : memref<1x3x5x7xf32>
// CHECK:         }
}

// -----

func.func @should_lower_to_zlow_unknown_dims(%arg0: tensor<1x?x?x7xf32>) -> tensor<*xf32> {
  %0 = "zhigh.Stick"(%arg0) {layout = "NHWC"} : (tensor<1x?x?x7xf32>) -> tensor<*xf16>
  %1 = "zhigh.Unstick"(%0) : (tensor<*xf16>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @should_lower_to_zlow_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?x7xf32>) -> memref<1x?x?x7xf32> {
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<1x?x?x7xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<1x?x?x7xf32>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<1x?x7x?xf16, [[MAP_0_]]>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_]]) {layout = "NCHW"} : (memref<1x?x?x7xf32>, memref<1x?x7x?xf16, [[MAP_0_]]>) -> ()
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[VAR_1_]], [[VAR_0_]]) {{.*}}: memref<1x?x?x7xf32>
// CHECK:           "zlow.unstick"([[RES_]], [[RES_1_]]) {layout = "NCHW"} : (memref<1x?x7x?xf16, [[MAP_0_]]>, memref<1x?x?x7xf32>) -> ()
// CHECK:           return [[RES_1_]] : memref<1x?x?x7xf32>
// CHECK:         }
}
