// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @conv_valid_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x3x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, %arg2: tensor<1xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<2x2x3x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, tensor<1xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2, d3) -> (d3 floordiv 64, d0, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
// CHECK-LABEL:  func @conv_valid_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x32x3xf16, #map>, [[PARAM_1_:%.+]]: memref<2x2x3x1xf16, #map1>, [[PARAM_2_:%.+]]: memref<1xf16, #map2>) -> memref<1x31x31x1xf16, #map> {
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_c31_i64_:%.+]] = arith.constant 31 : i64
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x31x31x1xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c31_i64_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c31_i64_]], [[RES_1_]]{{.}}[[VAR_c6_]]{{.}} : memref<7xi64>
// CHECK:           "zlow.conv2d"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[RES_1_]], [[RES_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (memref<1x32x32x3xf16, #map>, memref<2x2x3x1xf16, #map1>, memref<1xf16, #map2>, memref<7xi64>, memref<1x31x31x1xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x31x31x1xf16, #map>
// CHECK:         }
}

// -----

func.func @conv_same_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x3x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, %arg2: tensor<1xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<2x2x3x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, tensor<1xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2, d3) -> (d3 floordiv 64, d0, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
// CHECK-LABEL:  func @conv_same_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x32x3xf16, #map>, [[PARAM_1_:%.+]]: memref<2x2x3x1xf16, #map1>, [[PARAM_2_:%.+]]: memref<1xf16, #map2>) -> memref<1x32x32x1xf16, #map> {
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x32x32x1xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c6_]]{{.}} : memref<7xi64>
// CHECK:           "zlow.conv2d"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[RES_1_]], [[RES_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (memref<1x32x32x3xf16, #map>, memref<2x2x3x1xf16, #map1>, memref<1xf16, #map2>, memref<7xi64>, memref<1x32x32x1xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x32x32x1xf16, #map>
// CHECK:         }
}

// -----

func.func @conv_valid_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x?x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, %arg2: tensor<?xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<2x2x?x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, tensor<?xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2, d3) -> (d3 floordiv 64, d0, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
// CHECK-DAG: #map3 = affine_map<()[s0] -> (s0 - 1)>
// CHECK-LABEL:  func @conv_valid_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?x?xf16, #map>, [[PARAM_1_:%.+]]: memref<2x2x?x1xf16, #map1>, [[PARAM_2_:%.+]]: memref<?xf16, #map2>) -> memref<1x?x?x1xf16, #map> {
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c3_]] : memref<1x?x?x?xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map3(){{.}}[[VAR_0_]]{{.}}
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply #map3(){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_3_]], [[VAR_4_]]) {{.*}}: memref<1x?x?x1xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK:           krnl.store [[VAR_7_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_8_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_10_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK:           krnl.store [[VAR_10_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK:           krnl.store [[VAR_11_]], [[RES_1_]]{{.}}[[VAR_c6_]]{{.}} : memref<7xi64>
// CHECK:           "zlow.conv2d"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[RES_1_]], [[RES_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (memref<1x?x?x?xf16, #map>, memref<2x2x?x1xf16, #map1>, memref<?xf16, #map2>, memref<7xi64>, memref<1x?x?x1xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x?x?x1xf16, #map>
// CHECK:         }
}

// -----

func.func @conv_same_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x?x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, %arg2: tensor<?xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<2x2x?x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, tensor<?xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2, d3) -> (d3 floordiv 64, d0, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (0, d0 floordiv 64, 0, 0, 31, d0 mod 64)>
// CHECK-LABEL:  func @conv_same_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?x?xf16, #map>, [[PARAM_1_:%.+]]: memref<2x2x?x1xf16, #map1>, [[PARAM_2_:%.+]]: memref<?xf16, #map2>) -> memref<1x?x?x1xf16, #map> {
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c3_]] : memref<1x?x?x?xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<1x?x?x1xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK:           krnl.store [[VAR_5_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_6_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_7_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_8_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_c6_]]{{.}} : memref<7xi64>
// CHECK:           "zlow.conv2d"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[RES_1_]], [[RES_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (memref<1x?x?x?xf16, #map>, memref<2x2x?x1xf16, #map1>, memref<?xf16, #map2>, memref<7xi64>, memref<1x?x?x1xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x?x?x1xf16, #map>
// CHECK:         }
}

// -----

func.func @conv_same_padding_no_bias_unknown_dims(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x3x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>) -> tensor<*xf16> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %cst) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<2x2x3x1xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, none) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2, d3) -> (d3 floordiv 64, d0, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @conv_same_padding_no_bias_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x32x3xf16, #map>, [[PARAM_1_:%.+]]: memref<2x2x3x1xf16, #map1>) -> memref<1x32x32x1xf16, #map> {
// CHECK-DAG:       [[VAR_c6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x32x32x1xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<7xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c6_]]{{.}} : memref<7xi64>
// CHECK:           [[VAR_2_:%.+]] = "krnl.global"() {alignment = 4096 : i64, name = "constant_stickify_0", shape = [1, 1, 1, 1, 32, 64], value = dense_resource<zhigh> : tensor<4096xi8>} : () -> memref<1x1x1x1x32x64xf16>
// CHECK:           "zlow.conv2d"([[PARAM_0_]], [[PARAM_1_]], [[VAR_2_]], [[RES_1_]], [[RES_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (memref<1x32x32x3xf16, #map>, memref<2x2x3x1xf16, #map1>, memref<1x1x1x1x32x64xf16>, memref<7xi64>, memref<1x32x32x1xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x32x32x1xf16, #map>
// CHECK:         }
}
