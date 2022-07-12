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

func @test_einsum_diagonal(%arg0: tensor<3x3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<3x3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-LABEL:  func @test_einsum_diagonal
  // CHECK-SAME:   ([[PARAM_0:%.+]]: tensor<3x3xf32>) -> tensor<3xf32> {
  // CHECK-NEXT:      [[MASK:%.+]] = "onnx.Constant"() {value = dense<{{\[\[true, false, false\], \[false, true, false\], \[false, false, true\]\]}}> : tensor<3x3xi1>} : () -> tensor<3x3xi1>
  // CHECK-NEXT:      [[ZERO:%.+]] = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-NEXT:      [[WHER:%.+]] = "onnx.Where"([[MASK]], [[PARAM_0]], [[ZERO]]) : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
  // CHECK-NEXT:      [[AXES:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-NEXT:      [[RSUM:%.+]] = "onnx.ReduceSum"([[WHER]], [[AXES]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x3xf32>, tensor<1xi64>) -> tensor<3xf32>
  // CHECK-NEXT:      return [[RSUM]] : tensor<3xf32>
}
