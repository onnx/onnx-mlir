// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

// -----

func @test_einsum_transpose(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ji"} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>

  // CHECK-LABEL:  func @test_einsum_transpose
  // CHECK-SAME:   ([[PARAM_0:%.+]]: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // CHECK-NEXT:      [[RES:%.+]] = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-NEXT:      return [[RES]] : tensor<3x2xf32>
}
