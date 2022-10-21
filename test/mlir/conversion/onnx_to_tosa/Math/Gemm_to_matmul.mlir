// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-LABEL:  @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:  %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 3, 5]} : (tensor<3x5xf32>) -> tensor<1x3x5xf32>
// CHECK-DAG:  %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 5, 4]} : (tensor<5x4xf32>) -> tensor<1x5x4xf32>
// CHECK-DAG:  %[[AB:.*]] = "tosa.matmul"(%[[A]], %[[B]]) : (tensor<1x3x5xf32>, tensor<1x5x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:  %[[ABC:.*]] = "tosa.add"(%[[AB]], %arg2) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:  %[[RES:.*]] = "tosa.reshape"(%[[ABC]]) {new_shape = [3, 4]} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:  return %[[RES]] : tensor<3x4xf32>

// -----

func.func @test_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32>  {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.618 : f32} : (tensor<3x6xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-LABEL:  @test_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:    %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 3, 6]} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 6, 4]} : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
// CHECK-DAG:    %[[ALPHA:.*]] = "tosa.const"() {value = dense<1.618000e+00> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG:    %[[ALPHA_A:.*]] = "tosa.mul"(%[[ALPHA]], %[[A]]) {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[ALPHA_AB:.*]] = "tosa.matmul"(%[[ALPHA_A]], %[[B]]) : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[ALPHA_ABC:.*]] = "tosa.add"(%[[ALPHA_AB]], %arg2) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[RES:.*]] = "tosa.reshape"(%[[ALPHA_ABC]]) {new_shape = [3, 4]} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:    return %[[RES]] : tensor<3x4xf32>

// -----

func.func @test_beta(%arg0: tensor<3x6xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<3x6xf32>) -> tensor<3x6xf32> {
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

// -----

func.func @test_transA(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> { 
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transA = 1 : si64} : (tensor<6x3xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-LABEL:  @test_transA(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:    %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 6, 3]} : (tensor<6x3xf32>) -> tensor<1x6x3xf32>
// CHECK-DAG:    %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 6, 4]} : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK-DAG:    %[[TRANA:.*]] = "tosa.transpose"(%[[A]], %2) : (tensor<1x6x3xf32>, tensor<3xi32>) -> tensor<1x3x6xf32>
// CHECK-DAG:    %[[AB:.*]] = "tosa.matmul"(%[[TRANA]], %[[B]]) : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[ABC:.*]] = "tosa.add"(%[[AB]], %arg2) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:    %[[RES:.*]] = "tosa.reshape"(%[[ABC]]) {new_shape = [3, 4]} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:    return %6 : tensor<3x4xf32>

// -----
  
func.func @test_transB(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
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

// -----

func.func @test_no_C(%arg0: tensor<1x5xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x5xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {beta = 1.664 : f32, transB = 1 : si64} : (tensor<1x5xf32>, tensor<5x5xf32>, none) -> tensor<1x5xf32>
  return %0 : tensor<1x5xf32>
}
// CHECK-LABEL:  @test_no_C(%arg0: tensor<1x5xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x5xf32> {
// CHECK-DAG:    %[[A:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 1, 5]} : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
// CHECK-DAG:    %[[B:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 5, 5]} : (tensor<5x5xf32>) -> tensor<1x5x5xf32>
// CHECK-DAG:    "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () 
// CHECK-DAG:    %[[TRANB:.*]] = "tosa.transpose"(%[[B]], %3) : (tensor<1x5x5xf32>, tensor<3xi32>) -> tensor<1x5x5xf32>
// CHECK-DAG:    %[[AB:.*]] = "tosa.matmul"(%[[A]], %[[TRANB]]) : (tensor<1x1x5xf32>, tensor<1x5x5xf32>) -> tensor<1x1x5xf32>
// CHECK-DAG:    %[[RES:.*]] = "tosa.reshape"(%[[AB]]) {new_shape = [1, 5]} : (tensor<1x1x5xf32>) -> tensor<1x5xf32>
// CHECK-DAG:    return %[[RES]] : tensor<1x5xf32>

// -----

func.func @test_mixed(%arg0: tensor<11x5xf32>, %arg1: tensor<3x11xf32>, %arg2: tensor<5x3xf32>) -> tensor<5x3xf32> {
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
