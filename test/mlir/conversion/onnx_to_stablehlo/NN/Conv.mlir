// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s
func.func @test_convolution(%arg0 : tensor<1x1x5x5xf32>, %arg1 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %bias) : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
  "func.return"(%0) : (tensor<1x1x3x3xf32>) -> ()
// CHECK-LABEL: @test_convolution
// CHECK{LITERAL}: %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
}

func.func @test_convolution_with_padding(%arg0 : tensor<1x1x5x5xf32>, %arg1 : tensor<1x1x3x3xf32>) -> tensor<1x1x5x5xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %bias) {auto_pad = "NOTSET", kernel_shape = [3,3], pads = [1, 1, 1, 1]}: (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x5x5xf32>
  "func.return"(%0) : (tensor<1x1x5x5xf32>) -> ()
// CHECK-LABEL: @test_convolution_with_padding
// CHECK{LITERAL}: %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x5x5xf32>
}
