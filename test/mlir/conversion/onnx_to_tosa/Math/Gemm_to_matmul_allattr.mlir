// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_mixed(%arg0: tensor<11x5xf32>, %arg1: tensor<3x11xf32>, %arg2: tensor<5x3xf32>) -> tensor<5x3xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.402 : f32, beta = 1.998 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<11x5xf32>, tensor<3x11xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}
// CHECK-LABEL:  @test_mixed(%arg0: tensor<11x5xf32>, %arg1: tensor<3x11xf32>, %arg2: tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK-DAG:    %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 11, 5]} : (tensor<11x5xf32>) -> tensor<1x11x5xf32>
// CHECK-DAG:    %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 3, 11]} : (tensor<3x11xf32>) -> tensor<1x3x11xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-DAG:    %[[TRANA:.*]] = "tosa.transpose"(%[[A]], %2) : (tensor<1x11x5xf32>, tensor<3xi32>) -> tensor<1x5x11xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-DAG:    %[[TRANB:.*]] = "tosa.transpose"(%[[B]], %4) : (tensor<1x3x11xf32>, tensor<3xi32>) -> tensor<1x11x3xf32>
// CHECK-DAG:    %[[ALPHA:.*]] = "tosa.const"() {value = dense<1.402000e+00> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG:    %[[ALPHA_TA:.*]] = "tosa.mul"(%[[ALPHA]], %[[TRANA]]) {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x5x11xf32>) -> tensor<1x5x11xf32>
// CHECK-DAG:    %[[BETA:.*]] = "tosa.const"() {value = dense<1.998000e+00> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
// CHECK-DAG:    %[[BETA_C:.*]] = "tosa.mul"(%[[BETA]], %arg2) {shift = 0 : i32} : (tensor<1x1xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK-DAG:    %[[ALPHA_AB:.*]] = "tosa.matmul"(%[[ALPHA_TA]], %[[TRANB]]) : (tensor<1x5x11xf32>, tensor<1x11x3xf32>) -> tensor<1x5x3xf32>
// CHECK-DAG:    %[[ALPHA_AB_BETA_C:.*]] = "tosa.add"(%[[ALPHA_AB]], %[[BETA_C]]) : (tensor<1x5x3xf32>, tensor<5x3xf32>) -> tensor<1x5x3xf32>
// CHECK-DAG:    %[[RES:.*]] = "tosa.reshape"(%[[ALPHA_AB_BETA_C]]) {new_shape = [5, 3]} : (tensor<1x5x3xf32>) -> tensor<5x3xf32>
// CHECK-DAG:    return %[[RES]] : tensor<5x3xf32>