// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:  func.func @test_gemm_to_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x5xf32>, [[PARAM_1_:%.+]]: tensor<5x4xf32>, [[PARAM_2_:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 3, 5>} : (tensor<3x5xf32>) -> tensor<1x3x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 5, 4>} : (tensor<5x4xf32>) -> tensor<1x5x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.matmul [[VAR_0_]], [[VAR_1_]] : (tensor<1x3x5xf32>, tensor<1x5x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 3, 4>} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.add [[VAR_2_]], [[VAR_3_]] : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.reshape [[VAR_4_]] {new_shape = array<i64: 3, 4>} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_5_]] : tensor<3x4xf32>
// CHECK:         }
}

// -----

func.func @test_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32>  {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.618 : f32} : (tensor<3x6xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:  func.func @test_alpha
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x6xf32>, [[PARAM_1_:%.+]]: tensor<6x4xf32>, [[PARAM_2_:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 3, 6>} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 6, 4>} : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<1.618000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.mul [[VAR_2_]], [[VAR_0_]] {shift = 0 : i8} : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.matmul [[VAR_3_]], [[VAR_1_]] : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 3, 4>} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.add [[VAR_4_]], [[VAR_5_]] : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_6_]] {new_shape = array<i64: 3, 4>} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_7_]] : tensor<3x4xf32>
// CHECK:         }
}

// -----

func.func @test_beta(%arg0: tensor<3x6xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<3x6xf32>) -> tensor<3x6xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {beta = 1.349 : f32} : (tensor<3x6xf32>, tensor<6x6xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
  return %0 : tensor<3x6xf32>
// CHECK-LABEL:  func.func @test_beta
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x6xf32>, [[PARAM_1_:%.+]]: tensor<6x6xf32>, [[PARAM_2_:%.+]]: tensor<3x6xf32>) -> tensor<3x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 3, 6>} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 6, 6>} : (tensor<6x6xf32>) -> tensor<1x6x6xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<1.349000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 3, 6>} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.mul [[VAR_2_]], [[VAR_3_]] {shift = 0 : i8} : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.matmul [[VAR_0_]], [[VAR_1_]] : (tensor<1x3x6xf32>, tensor<1x6x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.add [[VAR_5_]], [[VAR_4_]] : (tensor<1x3x6xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_6_]] {new_shape = array<i64: 3, 6>} : (tensor<1x3x6xf32>) -> tensor<3x6xf32>
// CHECK:           return [[VAR_7_]] : tensor<3x6xf32>
// CHECK:         }
}

// -----

func.func @test_transa(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> { 
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transA = 1 : si64} : (tensor<6x3xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:  func.func @test_transa
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6x3xf32>, [[PARAM_1_:%.+]]: tensor<6x4xf32>, [[PARAM_2_:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 6, 3>} : (tensor<6x3xf32>) -> tensor<1x6x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 6, 4>} : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_0_]], [[VAR_2_]] : (tensor<1x6x3xf32>, tensor<3xi32>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.matmul [[VAR_3_]], [[VAR_1_]] : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 3, 4>} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.add [[VAR_4_]], [[VAR_5_]] : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_6_]] {new_shape = array<i64: 3, 4>} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_7_]] : tensor<3x4xf32>
// CHECK:         }
}

// -----
  
func.func @test_transb(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.184 : f32, transB = 1 : si64} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:  func.func @test_transb
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x6xf32>, [[PARAM_1_:%.+]]: tensor<4x6xf32>, [[PARAM_2_:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 3, 6>} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 4, 6>} : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_2_]] : (tensor<1x4x6xf32>, tensor<3xi32>) -> tensor<1x6x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<1.184000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.mul [[VAR_4_]], [[VAR_0_]] {shift = 0 : i8} : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.matmul [[VAR_5_]], [[VAR_3_]] : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 3, 4>} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_8_:%.+]] = tosa.add [[VAR_6_]], [[VAR_7_]] : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_9_:%.+]] = tosa.reshape [[VAR_8_]] {new_shape = array<i64: 3, 4>} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_9_]] : tensor<3x4xf32>
// CHECK:         }
}

// -----

func.func @test_no_c(%arg0: tensor<1x5xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x5xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {beta = 1.664 : f32, transB = 1 : si64} : (tensor<1x5xf32>, tensor<5x5xf32>, none) -> tensor<1x5xf32>
  return %0 : tensor<1x5xf32>
// CHECK-LABEL:  func.func @test_no_c
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5xf32>, [[PARAM_1_:%.+]]: tensor<5x5xf32>) -> tensor<1x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 1, 5>} : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 5, 5>} : (tensor<5x5xf32>) -> tensor<1x5x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x5x5xf32>, tensor<3xi32>) -> tensor<1x5x5xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.matmul [[VAR_1_]], [[VAR_4_]] : (tensor<1x1x5xf32>, tensor<1x5x5xf32>) -> tensor<1x1x5xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.reshape [[VAR_5_]] {new_shape = array<i64: 1, 5>} : (tensor<1x1x5xf32>) -> tensor<1x5xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x5xf32>
// CHECK:         }
}

// -----

func.func @test_no_c_no_trans(%arg0: tensor<1x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<1x6xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {alpha = 1.349 : f32} : (tensor<1x5xf32>, tensor<5x6xf32>, none) -> tensor<1x6xf32>
  return %0 : tensor<1x6xf32>
// CHECK-LABEL:  func.func @test_no_c_no_trans
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5xf32>, [[PARAM_1_:%.+]]: tensor<5x6xf32>) -> tensor<1x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 1, 5>} : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 5, 6>} : (tensor<5x6xf32>) -> tensor<1x5x6xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<1.349000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.mul [[VAR_3_]], [[VAR_1_]] {shift = 0 : i8} : (tensor<1x1x1xf32>, tensor<1x1x5xf32>) -> tensor<1x1x5xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.matmul [[VAR_4_]], [[VAR_2_]] : (tensor<1x1x5xf32>, tensor<1x5x6xf32>) -> tensor<1x1x6xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.reshape [[VAR_5_]] {new_shape = array<i64: 1, 6>} : (tensor<1x1x6xf32>) -> tensor<1x6xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x6xf32>
// CHECK:         }
}

// -----

func.func @test_mixed(%arg0: tensor<11x5xf32>, %arg1: tensor<3x11xf32>, %arg2: tensor<5x3xf32>) -> tensor<5x3xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.402 : f32, beta = 1.998 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<11x5xf32>, tensor<3x11xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
// CHECK-LABEL:  func.func @test_mixed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<11x5xf32>, [[PARAM_1_:%.+]]: tensor<3x11xf32>, [[PARAM_2_:%.+]]: tensor<5x3xf32>) -> tensor<5x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 11, 5>} : (tensor<11x5xf32>) -> tensor<1x11x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 3, 11>} : (tensor<3x11xf32>) -> tensor<1x3x11xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.transpose [[VAR_0_]], [[VAR_2_]] : (tensor<1x11x5xf32>, tensor<3xi32>) -> tensor<1x5x11xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_4_]] : (tensor<1x3x11xf32>, tensor<3xi32>) -> tensor<1x11x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<1.402000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.mul [[VAR_6_]], [[VAR_3_]] {shift = 0 : i8} : (tensor<1x1x1xf32>, tensor<1x5x11xf32>) -> tensor<1x5x11xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<1.998000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 5, 3>} : (tensor<5x3xf32>) -> tensor<1x5x3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.mul [[VAR_8_]], [[VAR_9_]] {shift = 0 : i8} : (tensor<1x1x1xf32>, tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.matmul [[VAR_7_]], [[VAR_5_]] : (tensor<1x5x11xf32>, tensor<1x11x3xf32>) -> tensor<1x5x3xf32>
// CHECK:           [[VAR_12_:%.+]] = tosa.add [[VAR_11_]], [[VAR_10_]] : (tensor<1x5x3xf32>, tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
// CHECK:           [[VAR_13_:%.+]] = tosa.reshape [[VAR_12_]] {new_shape = array<i64: 5, 3>} : (tensor<1x5x3xf32>) -> tensor<5x3xf32>
// CHECK:           return [[VAR_13_]] : tensor<5x3xf32>
// CHECK:         }
}