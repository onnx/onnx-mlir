// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_pad_constant(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x7x7xf32> {
  %0 = onnx.Constant dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>
  %1 = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.Pad"(%arg0, %0, %1, %2) {mode = "constant"} : (tensor<1x3x5x5xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x3x7x7xf32>
  return %3 : tensor<1x3x7x7xf32>
// CHECK-LABEL:  func.func @test_pad_constant(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x7x7xf32> {
// CHECK-NEXT:  %0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:  %1 = "stablehlo.pad"(%arg0, %0) {edge_padding_high = dense<[0, 0, 1, 1]> : vector<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : vector<4xi64>, interior_padding = dense<0> : vector<4xi64>} : (tensor<1x3x5x5xf32>, tensor<f32>) -> tensor<1x3x7x7xf32>
// CHECK-NEXT:  return %1 : tensor<1x3x7x7xf32>
}
