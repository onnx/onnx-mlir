 // RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s 

func.func @test_beta(%arg0: tensor<3x6xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<3x6xf32>) -> tensor<3x6xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {beta = 1.349 : f32} : (tensor<3x6xf32>, tensor<6x6xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
  return %0 : tensor<3x6xf32>
}
// CHECK-LABEL:  @test_beta(%arg0: tensor<3x6xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<3x6xf32>) -> tensor<3x6xf32>
// CHECK-DAG:    %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 3, 6]} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 6, 6]} : (tensor<6x6xf32>) -> tensor<1x6x6xf32>
// CHECK-DAG:    %[[BETA:.*]] = "tosa.const"() {value = dense<1.349000e+00> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
// CHECK-DAG:    %[[BETA_C:.*]] = "tosa.mul"(%[[BETA]], %arg2) {shift = 0 : i32} : (tensor<1x1xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
// CHECK-DAG:    %[[AB:.*]] = "tosa.matmul"(%[[A]], %[[B]]) : (tensor<1x3x6xf32>, tensor<1x6x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[AB_BETA_C:.*]] = "tosa.add"(%[[AB]], %[[BETA_C]]) : (tensor<1x3x6xf32>, tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[RES:.*]] = "tosa.reshape"(%[[AB_BETA_C]]) {new_shape = [3, 6]} : (tensor<1x3x6xf32>) -> tensor<3x6xf32>
// CHECK-DAG:    return %[[RES]] : tensor<3x6xf32>