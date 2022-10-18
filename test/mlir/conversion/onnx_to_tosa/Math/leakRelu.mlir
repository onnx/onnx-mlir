// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_leaky_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.707330704  : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
// CHECK-LABEL: test_leaky_relu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.707330704> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.mul"(%arg0, %[[VAR1]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.greater_equal"(%arg0, %[[VAR0]])
// CHECK: %[[VAR6:.*]] = "tosa.select"(%[[VAR3]], %arg0, %[[VAR2]])
}


