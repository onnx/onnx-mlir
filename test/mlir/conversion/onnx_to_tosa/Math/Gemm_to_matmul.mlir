// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:   func.func @test_gemm_to_matmul(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<3x5xf32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<5x4xf32>,
// CHECK-SAME:                                   %[[VAL_2:.*]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_0]]) <{new_shape = array<i64: 1, 3, 5>}> : (tensor<3x5xf32>) -> tensor<1x3x5xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_1]]) <{new_shape = array<i64: 1, 5, 4>}> : (tensor<5x4xf32>) -> tensor<1x5x4xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.matmul"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x3x5xf32>, tensor<1x5x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.add"(%[[VAL_5]], %[[VAL_2]]) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.reshape"(%[[VAL_6]]) <{new_shape = array<i64: 3, 4>}> : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return %[[VAL_7]] : tensor<3x4xf32>
// CHECK:         }
}

// -----

func.func @test_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32>  {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.618 : f32} : (tensor<3x6xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:   func.func @test_alpha(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<3x6xf32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: tensor<6x4xf32>,
// CHECK-SAME:                          %[[VAL_2:.*]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_0]]) <{new_shape = array<i64: 1, 3, 6>}> : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_1]]) <{new_shape = array<i64: 1, 6, 4>}> : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<1.618000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.mul"(%[[VAL_5]], %[[VAL_3]]) <{shift = 0 : i32}> : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.matmul"(%[[VAL_6]], %[[VAL_4]]) : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.add"(%[[VAL_7]], %[[VAL_2]]) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.reshape"(%[[VAL_8]]) <{new_shape = array<i64: 3, 4>}> : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return %[[VAL_9]] : tensor<3x4xf32>
// CHECK:         }
}

// -----

func.func @test_beta(%arg0: tensor<3x6xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<3x6xf32>) -> tensor<3x6xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {beta = 1.349 : f32} : (tensor<3x6xf32>, tensor<6x6xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
  return %0 : tensor<3x6xf32>
// CHECK-LABEL:   func.func @test_beta(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<3x6xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<6x6xf32>,
// CHECK-SAME:                         %[[VAL_2:.*]]: tensor<3x6xf32>) -> tensor<3x6xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_0]]) <{new_shape = array<i64: 1, 3, 6>}> : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_1]]) <{new_shape = array<i64: 1, 6, 6>}> : (tensor<6x6xf32>) -> tensor<1x6x6xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<1.349000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.mul"(%[[VAL_5]], %[[VAL_2]]) <{shift = 0 : i32}> : (tensor<1x1xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.matmul"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x3x6xf32>, tensor<1x6x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.add"(%[[VAL_7]], %[[VAL_6]]) : (tensor<1x3x6xf32>, tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.reshape"(%[[VAL_8]]) <{new_shape = array<i64: 3, 6>}> : (tensor<1x3x6xf32>) -> tensor<3x6xf32>
// CHECK:           return %[[VAL_9]] : tensor<3x6xf32>
// CHECK:         }
}

// -----

func.func @test_transa(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> { 
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transA = 1 : si64} : (tensor<6x3xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:   func.func @test_transa(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<6x3xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<6x4xf32>,
// CHECK-SAME:                           %[[VAL_2:.*]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_0]]) <{new_shape = array<i64: 1, 6, 3>}> : (tensor<6x3xf32>) -> tensor<1x6x3xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_1]]) <{new_shape = array<i64: 1, 6, 4>}> : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_6:.*]] = "tosa.transpose"(%[[VAL_3]], %[[VAL_5]]) : (tensor<1x6x3xf32>, tensor<3xi32>) -> tensor<1x3x6xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.matmul"(%[[VAL_6]], %[[VAL_4]]) : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.add"(%[[VAL_7]], %[[VAL_2]]) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.reshape"(%[[VAL_8]]) <{new_shape = array<i64: 3, 4>}> : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return %[[VAL_9]] : tensor<3x4xf32>
// CHECK:         }
}

// -----
  
func.func @test_transb(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.184 : f32, transB = 1 : si64} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:   func.func @test_transb(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<3x6xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<4x6xf32>,
// CHECK-SAME:                           %[[VAL_2:.*]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_0]]) <{new_shape = array<i64: 1, 3, 6>}> : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_1]]) <{new_shape = array<i64: 1, 4, 6>}> : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_6:.*]] = "tosa.transpose"(%[[VAL_4]], %[[VAL_5]]) : (tensor<1x4x6xf32>, tensor<3xi32>) -> tensor<1x6x4xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{value = dense<1.184000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.mul"(%[[VAL_7]], %[[VAL_3]]) <{shift = 0 : i32}> : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.matmul"(%[[VAL_8]], %[[VAL_6]]) : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           %[[VAL_10:.*]] = "tosa.add"(%[[VAL_9]], %[[VAL_2]]) : (tensor<1x3x4xf32>, tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           %[[VAL_11:.*]] = "tosa.reshape"(%[[VAL_10]]) <{new_shape = array<i64: 3, 4>}> : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return %[[VAL_11]] : tensor<3x4xf32>
// CHECK:         }
}

// -----

func.func @test_no_c(%arg0: tensor<1x5xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x5xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {beta = 1.664 : f32, transB = 1 : si64} : (tensor<1x5xf32>, tensor<5x5xf32>, none) -> tensor<1x5xf32>
  return %0 : tensor<1x5xf32>
// CHECK-LABEL:   func.func @test_no_c(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<1x5xf32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: tensor<5x5xf32>) -> tensor<1x5xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_0]]) <{new_shape = array<i64: 1, 1, 5>}> : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_1]]) <{new_shape = array<i64: 1, 5, 5>}> : (tensor<5x5xf32>) -> tensor<1x5x5xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_6:.*]] = "tosa.transpose"(%[[VAL_4]], %[[VAL_5]]) : (tensor<1x5x5xf32>, tensor<3xi32>) -> tensor<1x5x5xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.matmul"(%[[VAL_3]], %[[VAL_6]]) : (tensor<1x1x5xf32>, tensor<1x5x5xf32>) -> tensor<1x1x5xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.reshape"(%[[VAL_7]]) <{new_shape = array<i64: 1, 5>}> : (tensor<1x1x5xf32>) -> tensor<1x5xf32>
// CHECK:           return %[[VAL_8]] : tensor<1x5xf32>
// CHECK:         }
}

// -----

func.func @test_no_c_no_trans(%arg0: tensor<1x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<1x6xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {alpha = 1.349 : f32} : (tensor<1x5xf32>, tensor<5x6xf32>, none) -> tensor<1x6xf32>
  return %0 : tensor<1x6xf32>
// CHECK-LABEL:   func.func @test_no_c_no_trans(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<1x5xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<5x6xf32>) -> tensor<1x6xf32> {
// CHECK:           %[[VAL_2:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_0]]) <{new_shape = array<i64: 1, 1, 5>}> : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_1]]) <{new_shape = array<i64: 1, 5, 6>}> : (tensor<5x6xf32>) -> tensor<1x5x6xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<1.349000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "tosa.mul"(%[[VAL_5]], %[[VAL_3]]) <{shift = 0 : i32}> : (tensor<1x1x1xf32>, tensor<1x1x5xf32>) -> tensor<1x1x5xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.matmul"(%[[VAL_6]], %[[VAL_4]]) : (tensor<1x1x5xf32>, tensor<1x5x6xf32>) -> tensor<1x1x6xf32>
// CHECK:           %[[VAL_8:.*]] = "tosa.reshape"(%[[VAL_7]]) <{new_shape = array<i64: 1, 6>}> : (tensor<1x1x6xf32>) -> tensor<1x6xf32>
// CHECK:           return %[[VAL_8]] : tensor<1x6xf32>
// CHECK:         }
}

// -----

func.func @test_mixed(%arg0: tensor<11x5xf32>, %arg1: tensor<3x11xf32>, %arg2: tensor<5x3xf32>) -> tensor<5x3xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.402 : f32, beta = 1.998 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<11x5xf32>, tensor<3x11xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
// CHECK-LABEL:   func.func @test_mixed(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<11x5xf32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: tensor<3x11xf32>,
// CHECK-SAME:                          %[[VAL_2:.*]]: tensor<5x3xf32>) -> tensor<5x3xf32> {
// CHECK:           %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_0]]) <{new_shape = array<i64: 1, 11, 5>}> : (tensor<11x5xf32>) -> tensor<1x11x5xf32>
// CHECK:           %[[VAL_4:.*]] = "tosa.reshape"(%[[VAL_1]]) <{new_shape = array<i64: 1, 3, 11>}> : (tensor<3x11xf32>) -> tensor<1x3x11xf32>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_6:.*]] = "tosa.transpose"(%[[VAL_3]], %[[VAL_5]]) : (tensor<1x11x5xf32>, tensor<3xi32>) -> tensor<1x5x11xf32>
// CHECK:           %[[VAL_7:.*]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           %[[VAL_8:.*]] = "tosa.transpose"(%[[VAL_4]], %[[VAL_7]]) : (tensor<1x3x11xf32>, tensor<3xi32>) -> tensor<1x11x3xf32>
// CHECK:           %[[VAL_9:.*]] = "tosa.const"() <{value = dense<1.402000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           %[[VAL_10:.*]] = "tosa.mul"(%[[VAL_9]], %[[VAL_6]]) <{shift = 0 : i32}> : (tensor<1x1x1xf32>, tensor<1x5x11xf32>) -> tensor<1x5x11xf32>
// CHECK:           %[[VAL_11:.*]] = "tosa.const"() <{value = dense<1.998000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:           %[[VAL_12:.*]] = "tosa.mul"(%[[VAL_11]], %[[VAL_2]]) <{shift = 0 : i32}> : (tensor<1x1xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK:           %[[VAL_13:.*]] = "tosa.matmul"(%[[VAL_10]], %[[VAL_8]]) : (tensor<1x5x11xf32>, tensor<1x11x3xf32>) -> tensor<1x5x3xf32>
// CHECK:           %[[VAL_14:.*]] = "tosa.add"(%[[VAL_13]], %[[VAL_12]]) : (tensor<1x5x3xf32>, tensor<5x3xf32>) -> tensor<1x5x3xf32>
// CHECK:           %[[VAL_15:.*]] = "tosa.reshape"(%[[VAL_14]]) <{new_shape = array<i64: 5, 3>}> : (tensor<1x5x3xf32>) -> tensor<5x3xf32>
// CHECK:           return %[[VAL_15]] : tensor<5x3xf32>
// CHECK:         }
}