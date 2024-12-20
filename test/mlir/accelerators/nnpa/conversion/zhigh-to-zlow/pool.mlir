// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @maxpool_valid_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @maxpool_valid_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x32x3xf16, #map>) -> memref<1x31x31x3xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c31_i64_:%.+]] = arith.constant 31 : i64
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x31x31x3xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c31_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c31_i64_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.maxpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (memref<1x32x32x3xf16, #map>, memref<6xi64>, memref<1x31x31x3xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x31x31x3xf16, #map>
// CHECK:         }
}

// -----

func.func @maxpool_same_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @maxpool_same_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x32x3xf16, #map>) -> memref<1x32x32x3xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x32x32x3xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.maxpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (memref<1x32x32x3xf16, #map>, memref<6xi64>, memref<1x32x32x3xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x32x32x3xf16, #map>
// CHECK:         }
}

// -----

func.func @maxpool_valid_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 - 1)>
// CHECK-LABEL:  func @maxpool_valid_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?x?xf16, #map>) -> memref<1x?x?x?xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c3_]] : memref<1x?x?x?xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map1(){{.}}[[VAR_0_]]{{.}}
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply #map1(){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_3_]], [[VAR_4_]], [[VAR_2_]]) {{.*}}: memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK:           krnl.store [[VAR_7_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_8_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_10_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK:           krnl.store [[VAR_10_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK:           krnl.store [[VAR_11_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.maxpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (memref<1x?x?x?xf16, #map>, memref<6xi64>, memref<1x?x?x?xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x?x?x?xf16, #map>
// CHECK:         }
}

// -----

func.func @maxpool_same_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @maxpool_same_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?x?xf16, #map>) -> memref<1x?x?x?xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c3_]] : memref<1x?x?x?xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {{.*}}: memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK:           krnl.store [[VAR_5_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_6_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_7_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_8_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.maxpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (memref<1x?x?x?xf16, #map>, memref<6xi64>, memref<1x?x?x?xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x?x?x?xf16, #map>
// CHECK:         }
}

// -----

func.func @maxpool_same_padding_no_bias_unknown_dims(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @maxpool_same_padding_no_bias_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x32x3xf16, #map>) -> memref<1x32x32x3xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x32x32x3xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.maxpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (memref<1x32x32x3xf16, #map>, memref<6xi64>, memref<1x32x32x3xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x32x32x3xf16, #map>
// CHECK:         }
}

// -----

func.func @avgpool_valid_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @avgpool_valid_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x32x3xf16, #map>) -> memref<1x31x31x3xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c31_i64_:%.+]] = arith.constant 31 : i64
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x31x31x3xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c31_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c31_i64_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.avgpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (memref<1x32x32x3xf16, #map>, memref<6xi64>, memref<1x31x31x3xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x31x31x3xf16, #map>
// CHECK:         }
}

// -----

func.func @avgpool_same_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @avgpool_same_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x32x3xf16, #map>) -> memref<1x32x32x3xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x32x32x3xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.avgpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (memref<1x32x32x3xf16, #map>, memref<6xi64>, memref<1x32x32x3xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x32x32x3xf16, #map>
// CHECK:         }
}

// -----

func.func @avgpool_valid_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 - 1)>
// CHECK-LABEL:  func @avgpool_valid_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?x?xf16, #map>) -> memref<1x?x?x?xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c3_]] : memref<1x?x?x?xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #map1(){{.}}[[VAR_0_]]{{.}}
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply #map1(){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_3_]], [[VAR_4_]], [[VAR_2_]]) {{.*}}: memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK:           krnl.store [[VAR_7_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_8_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_10_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK:           krnl.store [[VAR_10_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK:           krnl.store [[VAR_11_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.avgpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (memref<1x?x?x?xf16, #map>, memref<6xi64>, memref<1x?x?x?xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x?x?x?xf16, #map>
// CHECK:         }
}

// -----

func.func @avgpool_same_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @avgpool_same_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x?x?x?xf16, #map>) -> memref<1x?x?x?xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c3_]] : memref<1x?x?x?xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {{.*}}: memref<1x?x?x?xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK:           krnl.store [[VAR_5_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_6_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_7_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_8_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.avgpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (memref<1x?x?x?xf16, #map>, memref<6xi64>, memref<1x?x?x?xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x?x?x?xf16, #map>
// CHECK:         }
}

// -----

func.func @avgpool_same_padding_no_bias_unknown_dims(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-DAG: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @avgpool_same_padding_no_bias_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x32x32x3xf16, #map>) -> memref<1x32x32x3xf16, #map> {
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x32x32x3xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<6xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<6xi64>
// CHECK:           krnl.store [[VAR_c32_i64_]], [[RES_1_]]{{.}}[[VAR_c5_]]{{.}} : memref<6xi64>
// CHECK:           "zlow.avgpool2d"([[PARAM_0_]], [[RES_1_]], [[RES_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (memref<1x32x32x3xf16, #map>, memref<6xi64>, memref<1x32x32x3xf16, #map>) -> ()
// CHECK:           return [[RES_]] : memref<1x32x32x3xf16, #map>
// CHECK:         }
}
