// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s
module attributes {}  {
  func.func @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
// CHECK-LABEL:  @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
// CHECK-DAG:  %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 3, 5]} : (tensor<3x5xf32>) -> tensor<1x3x5xf32>
// CHECK-DAG:  %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 5, 4]} : (tensor<5x4xf32>) -> tensor<1x5x4xf32>
// CHECK-DAG:  %[[AB:.*]] = "tosa.matmul"(%[[A]], %[[B]]) : (tensor<1x3x5xf32>, tensor<1x5x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:  %[[ABC:.*]] = "tosa.add"(%[[AB]], %arg2) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:  %[[RES:.*]] = "tosa.reshape"(%[[ABC]]) {new_shape = [3, 4]} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:  return %[[RES]] : tensor<3x4xf32>
}
