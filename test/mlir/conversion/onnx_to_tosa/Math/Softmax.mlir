// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.Softmax"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK-LABEL: test_softmax
// CHECK-DAG: %[[VAR0:.*]] = "tosa.exp"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reciprocal"(%[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR2]]) {shift = 0 : i32}
}
