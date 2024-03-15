// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize -split-input-file %s | FileCheck %s
func.func @test_grouped(%arg0 : tensor<1x72x8x14xf32>, %arg1 : tensor<72x24x4x4xf32>, %arg2 : tensor<72xf32>) -> tensor<1x72x16x28xf32> {
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %arg2) {group = 3 : si64, kernel_shape = [4, 4], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x72x8x14xf32>, tensor<72x24x4x4xf32>, tensor<72xf32>) -> tensor<1x72x16x28xf32>
  "func.return"(%0) : (tensor<1x72x16x28xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_grouped
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x72x8x14xf32>, [[PARAM_1_:%.+]]: tensor<72x24x4x4xf32>, [[PARAM_2_:%.+]]: tensor<72xf32>) -> tensor<1x72x16x28xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [1, 72, 16, 28] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.reverse [[PARAM_1_]], dims = [2, 3] : tensor<72x24x4x4xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.reshape [[VAR_1_]] : (tensor<72x24x4x4xf32>) -> tensor<3x24x24x4x4xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.transpose [[VAR_2_]], dims = [0, 2, 1, 3, 4] : (tensor<3x24x24x4x4xf32>) -> tensor<3x24x24x4x4xf32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.reshape [[VAR_3_]] : (tensor<3x24x24x4x4xf32>) -> tensor<72x24x4x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.convolution([[PARAM_0_]], [[VAR_4_]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = {{.}}[2, 2], [2, 2]{{.}}, lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<1x72x8x14xf32>, tensor<72x24x4x4xf32>) -> tensor<1x72x16x28xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_2_]], [[VAR_0_]], dims = [1] : (tensor<72xf32>, tensor<4xindex>) -> tensor<1x72x16x28xf32>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.add [[VAR_5_]], [[VAR_6_]] : tensor<1x72x16x28xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x72x16x28xf32>
// CHECK:         }

// -----

func.func @test_dynamic_shape(%arg0 : tensor<?x2x3x3xf32>, %arg1 : tensor<2x2x3x3xf32>) -> tensor<?x2x5x5xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) : (tensor<?x2x3x3xf32>, tensor<2x2x3x3xf32>, none) -> tensor<?x2x5x5xf32>
  "func.return"(%0) : (tensor<?x2x5x5xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_dynamic_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x2x3x3xf32>, [[PARAM_1_:%.+]]: tensor<2x2x3x3xf32>) -> tensor<?x2x5x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.reverse [[PARAM_1_]], dims = [2, 3] : tensor<2x2x3x3xf32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.transpose [[VAR_0_]], dims = [1, 0, 2, 3] : (tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.convolution([[PARAM_0_]], [[VAR_1_]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = {{.}}[2, 2], [2, 2]{{.}}, lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x2x3x3xf32>, tensor<2x2x3x3xf32>) -> tensor<?x2x5x5xf32>
// CHECK:           return [[VAR_2_]] : tensor<?x2x5x5xf32>
// CHECK:         }

// -----

func.func @test_valid(%arg0 : tensor<1x2x3x3xf32>, %arg1 : tensor<2x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {auto_pad = "VALID"} : (tensor<1x2x3x3xf32>, tensor<2x2x3x3xf32>, none) -> tensor<1x2x5x5xf32>
  "func.return"(%0) : (tensor<1x2x5x5xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_valid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3x3xf32>, [[PARAM_1_:%.+]]: tensor<2x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.reverse [[PARAM_1_]], dims = [2, 3] : tensor<2x2x3x3xf32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.transpose [[VAR_0_]], dims = [1, 0, 2, 3] : (tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.convolution([[PARAM_0_]], [[VAR_1_]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = {{.}}[2, 2], [2, 2]{{.}}, lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x3x3xf32>, tensor<2x2x3x3xf32>) -> tensor<1x2x5x5xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x2x5x5xf32>
// CHECK:         }

// -----

func.func @test_attributes_0(%arg0 : tensor<1x1x3x3xf32>, %arg1 : tensor<1x2x3x3xf32>) -> tensor<1x2x9x7xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {strides = [3, 2], output_shape = [9, 7]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x9x7xf32>
  "func.return"(%0) : (tensor<1x2x9x7xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_attributes_0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x9x7xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.reverse [[PARAM_1_]], dims = [2, 3] : tensor<1x2x3x3xf32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.transpose [[VAR_0_]], dims = [1, 0, 2, 3] : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.convolution([[PARAM_0_]], [[VAR_1_]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = {{.}}[2, 2], [2, 2]{{.}}, lhs_dilate = [3, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x3xf32>, tensor<2x1x3x3xf32>) -> tensor<1x2x9x7xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x2x9x7xf32>
// CHECK:         }

// -----

func.func @test_attributes_1(%arg0 : tensor<?x1x3x3xf32>, %arg1 : tensor<1x2x3x3xf32>) -> tensor<?x2x10x8xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {strides = [3, 2], output_padding = [1, 1]} : (tensor<?x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<?x2x10x8xf32>
  "func.return"(%0) : (tensor<?x2x10x8xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_attributes_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<?x2x10x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.reverse [[PARAM_1_]], dims = [2, 3] : tensor<1x2x3x3xf32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.transpose [[VAR_0_]], dims = [1, 0, 2, 3] : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.convolution([[PARAM_0_]], [[VAR_1_]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = {{.}}[2, 3], [2, 3]{{.}}, lhs_dilate = [3, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x1x3x3xf32>, tensor<2x1x3x3xf32>) -> tensor<?x2x10x8xf32>
// CHECK:           return [[VAR_2_]] : tensor<?x2x10x8xf32>
// CHECK:         }

// -----

func.func @test_dilations(%arg0 : tensor<1x1x3x3xf32>, %arg1 : tensor<1x1x2x2xf32>) -> tensor<1x1x5x5xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {dilations = [2, 2]} : (tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>, none) -> tensor<1x1x5x5xf32>
  "func.return"(%0) : (tensor<1x1x5x5xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_dilations
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x1x2x2xf32>) -> tensor<1x1x5x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.reverse [[PARAM_1_]], dims = [2, 3] : tensor<1x1x2x2xf32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.transpose [[VAR_0_]], dims = [1, 0, 2, 3] : (tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.convolution([[PARAM_0_]], [[VAR_1_]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = {{.}}[2, 2], [2, 2]{{.}}, lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x5x5xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x1x5x5xf32>
// CHECK:         }

// -----

func.func @test_pads(%arg0 : tensor<1x1x3x3xf32>, %arg1 : tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {strides = [3, 2], pads = [1, 2, 1, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x7x3xf32>
  "func.return"(%0) : (tensor<1x2x7x3xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_pads
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.reverse [[PARAM_1_]], dims = [2, 3] : tensor<1x2x3x3xf32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.transpose [[VAR_0_]], dims = [1, 0, 2, 3] : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.convolution([[PARAM_0_]], [[VAR_1_]]) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = {{.}}[1, 1], [0, 0]{{.}}, lhs_dilate = [3, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x3xf32>, tensor<2x1x3x3xf32>) -> tensor<1x2x7x3xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x2x7x3xf32>
// CHECK:         }
