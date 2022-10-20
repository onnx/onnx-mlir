// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_no_C(%arg0: tensor<1x5xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x5xf32> attributes {input_names = ["a", "b"], output_names = ["y"]} {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {beta = 1.664 : f32, transB = 1 : si64} : (tensor<1x5xf32>, tensor<5x5xf32>, none) -> tensor<1x5xf32>
  return %0 : tensor<1x5xf32>
}
// CHECK-LABEL:  @test_no_C(%arg0: tensor<1x5xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x5xf32> attributes {input_names = ["a", "b"], output_names = ["y"]} {
// CHECK-NOT:    %none = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:    %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 1, 5]} : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
// CHECK-DAG:    %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 5, 5]} : (tensor<5x5xf32>) -> tensor<1x5x5xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () 
// CHECK-DAG:    %[[TRANB:.*]] = "tosa.transpose"(%[[B]], %2) : (tensor<1x5x5xf32>, tensor<3xi32>) -> tensor<1x5x5xf32>
// CHECK-DAG:    %[[AB:.*]] = "tosa.matmul"(%[[A]], %[[TRANB]]) : (tensor<1x1x5xf32>, tensor<1x5x5xf32>) -> tensor<1x1x5xf32>
// CHECK-DAG:    %[[RES:.*]] = "tosa.reshape"(%[[AB]]) {new_shape = [1, 5]} : (tensor<1x1x5xf32>) -> tensor<1x5xf32>
// CHECK-DAG:    return %[[RES]] : tensor<1x5xf32>