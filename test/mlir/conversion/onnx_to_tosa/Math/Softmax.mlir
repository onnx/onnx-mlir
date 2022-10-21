// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.Softmax"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK-LABEL: test_softmax
// CHECK: %[[VAR0:.*]] = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 2 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x1xf32>
// CHECK: %[[VAR2:.*]] = "tosa.reciprocal"(%[[VAR1]]) : (tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
// CHECK: %[[VAR3:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR2]]) {shift = 0 : i32}  : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
}

// -----
func.func @test_axis_one_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK-LABEL: test_axis_one_softmax
// CHECK: %[[VAR0:.*]] = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 1 : i64} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
// CHECK: %[[VAR2:.*]] = "tosa.reciprocal"(%[[VAR1]]) : (tensor<13x1x3xf32>) -> tensor<13x1x3xf32>
// CHECK: %[[VAR3:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR2]]) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
}
