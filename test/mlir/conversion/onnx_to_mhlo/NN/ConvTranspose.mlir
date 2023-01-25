// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --canonicalize -split-input-file %s | FileCheck %s
func.func @test_grouped(%arg0 : tensor<1x72x8x14xf32>, %arg1 : tensor<72x24x4x4xf32>, %arg2 : tensor<72xf32>) -> tensor<1x72x16x28xf32> {
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %arg2) {group = 3 : si64, kernel_shape = [4, 4], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x72x8x14xf32>, tensor<72x24x4x4xf32>, tensor<72xf32>) -> tensor<1x72x16x28xf32>
  "func.return"(%0) : (tensor<1x72x16x28xf32>) -> ()
// CHECK-LABEL: @test_grouped
// CHECK{LITERAL}: %0 = "mhlo.reverse"(%arg1) {dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<72x24x4x4xf32>) -> tensor<72x24x4x4xf32>
// CHECK{LITERAL}: %1 = mhlo.reshape %0 : (tensor<72x24x4x4xf32>) -> tensor<3x24x24x4x4xf32>
// CHECK{LITERAL}: %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 2, 1, 3, 4]> : tensor<5xi64>} : (tensor<3x24x24x4x4xf32>) -> tensor<3x24x24x4x4xf32>
// CHECK{LITERAL}: %3 = mhlo.reshape %2 : (tensor<3x24x24x4x4xf32>) -> tensor<72x24x4x4xf32>
// CHECK{LITERAL}: %4 = mhlo.convolution(%arg0, %3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<1x72x8x14xf32>, tensor<72x24x4x4xf32>) -> tensor<1x72x16x28xf32>
// CHECK{LITERAL}: %5 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<72xf32>) -> tensor<1x72x16x28xf32>
// CHECK{LITERAL}: %6 = mhlo.add %4, %5 : tensor<1x72x16x28xf32>
// CHECK{LITERAL}: return %6 : tensor<1x72x16x28xf32>
}

func.func @test_dynamic_shape(%arg0 : tensor<?x2x3x3xf32>, %arg1 : tensor<2x2x3x3xf32>) -> tensor<?x2x5x5xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) : (tensor<?x2x3x3xf32>, tensor<2x2x3x3xf32>, none) -> tensor<?x2x5x5xf32>
  "func.return"(%0) : (tensor<?x2x5x5xf32>) -> ()
// CHECK-LABEL: @test_dynamic_shape
// CHECK{LITERAL}: %0 = "mhlo.reverse"(%arg1) {dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>
// CHECK{LITERAL}: %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 2, 3]> : tensor<4xi64>} : (tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>
// CHECK{LITERAL}: %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x2x3x3xf32>, tensor<2x2x3x3xf32>) -> tensor<?x2x5x5xf32>
// CHECK{LITERAL}: return %2 : tensor<?x2x5x5xf32>
}

func.func @test_valid(%arg0 : tensor<1x2x3x3xf32>, %arg1 : tensor<2x2x3x3xf32>) -> tensor<1x2x5x5xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {auto_pad = "VALID"} : (tensor<1x2x3x3xf32>, tensor<2x2x3x3xf32>, none) -> tensor<1x2x5x5xf32>
  "func.return"(%0) : (tensor<1x2x5x5xf32>) -> ()
// CHECK-LABEL: @test_valid
// CHECK{LITERAL}: %0 = "mhlo.reverse"(%arg1) {dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>
// CHECK{LITERAL}: %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 2, 3]> : tensor<4xi64>} : (tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>
// CHECK{LITERAL}: %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x3x3xf32>, tensor<2x2x3x3xf32>) -> tensor<1x2x5x5xf32>
// CHECK{LITERAL}: return %2 : tensor<1x2x5x5xf32>
}

func.func @test_attributes_0(%arg0 : tensor<1x1x3x3xf32>, %arg1 : tensor<1x2x3x3xf32>) -> tensor<1x2x9x7xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {strides = [3, 2], output_shape = [9, 7]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x9x7xf32>
  "func.return"(%0) : (tensor<1x2x9x7xf32>) -> ()
// CHECK-LABEL: @test_attributes_0
// CHECK{LITERAL}: %0 = "mhlo.reverse"(%arg1) {dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
// CHECK{LITERAL}: %1 = mhlo.reshape %0 : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK{LITERAL}: %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [3, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x3xf32>, tensor<2x1x3x3xf32>) -> tensor<1x2x9x7xf32>
// CHECK{LITERAL}: return %2 : tensor<1x2x9x7xf32>
}

func.func @test_attributes_1(%arg0 : tensor<?x1x3x3xf32>, %arg1 : tensor<1x2x3x3xf32>) -> tensor<?x2x10x8xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {strides = [3, 2], output_padding = [1, 1]} : (tensor<?x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<?x2x10x8xf32>
  "func.return"(%0) : (tensor<?x2x10x8xf32>) -> ()
// CHECK-LABEL: @test_attributes_1
// CHECK{LITERAL}: %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK{LITERAL}: %1 = "mhlo.reverse"(%arg1) {dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
// CHECK{LITERAL}: %2 = mhlo.reshape %1 : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK{LITERAL}: %3 = mhlo.convolution(%arg0, %2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [3, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x1x3x3xf32>, tensor<2x1x3x3xf32>) -> tensor<?x2x9x7xf32>
// CHECK{LITERAL}: %4 = "mhlo.pad"(%3, %0) {edge_padding_high = dense<[0, 0, 1, 1]> : vector<4xi64>, edge_padding_low = dense<0> : vector<4xi64>, interior_padding = dense<0> : vector<4xi64>} : (tensor<?x2x9x7xf32>, tensor<f32>) -> tensor<?x2x10x8xf32>
// CHECK{LITERAL}: return %4 : tensor<?x2x10x8xf32>
}

func.func @test_dilations(%arg0 : tensor<1x1x3x3xf32>, %arg1 : tensor<1x1x2x2xf32>) -> tensor<1x1x5x5xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {dilations = [2, 2]} : (tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>, none) -> tensor<1x1x5x5xf32>
  "func.return"(%0) : (tensor<1x1x5x5xf32>) -> ()
// CHECK-LABEL: @test_dilations
// CHECK{LITERAL}: %0 = "mhlo.reverse"(%arg1) {dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
// CHECK{LITERAL}: %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x5x5xf32>
// CHECK{LITERAL}: return %1 : tensor<1x1x5x5xf32>
}

func.func @test_pads(%arg0 : tensor<1x1x3x3xf32>, %arg1 : tensor<1x2x3x3xf32>) -> tensor<1x2x7x3xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %bias) {strides = [3, 2], pads = [1, 2, 1, 2]} : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, none) -> tensor<1x2x7x3xf32>
  "func.return"(%0) : (tensor<1x2x7x3xf32>) -> ()
// CHECK-LABEL: @test_pads
// CHECK{LITERAL}: %0 = "mhlo.reverse"(%arg1) {dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
// CHECK{LITERAL}: %1 = mhlo.reshape %0 : (tensor<1x2x3x3xf32>) -> tensor<2x1x3x3xf32>
// CHECK{LITERAL}: %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [0, 0]], lhs_dilate = [3, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x3xf32>, tensor<2x1x3x3xf32>) -> tensor<1x2x7x3xf32>
// CHECK{LITERAL}: return %2 : tensor<1x2x7x3xf32>
}
