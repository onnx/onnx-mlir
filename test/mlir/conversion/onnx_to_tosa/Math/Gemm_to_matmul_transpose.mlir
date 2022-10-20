// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_transA(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transA = 1 : si64} : (tensor<6x3xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-LABEL:  @test_transA(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
// CHECK-DAG:    %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 6, 3]} : (tensor<6x3xf32>) -> tensor<1x6x3xf32>
// CHECK-DAG:    %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 6, 4]} : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-DAG:    %[[TRANA:.*]] = "tosa.transpose"(%[[A]], %2) : (tensor<1x6x3xf32>, tensor<3xi32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[AB:.*]] = "tosa.matmul"(%[[TRANA]], %[[B]]) : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[ABC:.*]] = "tosa.add"(%[[AB]], %arg2) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[RES:.*]] = "tosa.reshape"(%[[ABC]]) {new_shape = [3, 4]} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:    return %6 : tensor<3x4xf32>
  
func.func @test_transB(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.184 : f32, transB = 1 : si64} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-LABEL:  @test_transB(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:    %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 3, 6]} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 4, 6]} : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-DAG:    %[[TRANB:.*]] = "tosa.transpose"(%[[B]], %2) : (tensor<1x4x6xf32>, tensor<3xi32>) -> tensor<1x6x4xf32>
// CHECK-DAG:    %[[ALPHA:.*]] = "tosa.const"() {value = dense<1.184000e+00> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG:    %[[ALPHA_A:.*]] = "tosa.mul"(%[[ALPHA]], %[[A]]) {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[ALPHA_AB:.*]] = "tosa.matmul"(%[[ALPHA_A]], %[[TRANB]]) : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[ALPHA_ABC:.*]] = "tosa.add"(%[[ALPHA_AB]], %arg2) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[RES:.*]] = "tosa.reshape"(%[[ALPHA_ABC]]) {new_shape = [3, 4]} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:    return %[[RES]] : tensor<3x4xf32>