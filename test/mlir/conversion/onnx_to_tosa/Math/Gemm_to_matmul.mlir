// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:  func.func @test_gemm_to_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x5xf32>, [[PARAM_1_:%.+]]: tensor<5x4xf32>, [[PARAM_2_:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:       [[SHAPE_0:%.+]] = tosa.const_shape {value = dense<[1, 3, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_0]] : (tensor<3x5xf32>, !tosa.shape<3>) -> tensor<1x3x5xf32>
// CHECK-DAG:       [[SHAPE_1:%.+]] = tosa.const_shape {value = dense<[1, 5, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_1]] : (tensor<5x4xf32>, !tosa.shape<3>) -> tensor<1x5x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_2_:%.+]] = tosa.matmul [[VAR_0_]], [[VAR_1_]] : (tensor<1x3x5xf32>, tensor<1x5x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[SHAPE_2:%.+]] = tosa.const_shape {value = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_3_:%.+]] = tosa.reshape [[PARAM_2_]], [[SHAPE_2]] : (tensor<3x4xf32>, !tosa.shape<3>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.add [[VAR_2_]], [[VAR_3_]] : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[SHAPE_3:%.+]] = tosa.const_shape {value = dense<[3, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_5_:%.+]] = tosa.reshape [[VAR_4_]], [[SHAPE_3]] : (tensor<1x3x4xf32>, !tosa.shape<2>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_5_]] : tensor<3x4xf32>
// CHECK:         }
}

// -----

func.func @test_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32>  {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.618 : f32} : (tensor<3x6xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:  func.func @test_alpha
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x6xf32>, [[PARAM_1_:%.+]]: tensor<6x4xf32>, [[PARAM_2_:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:       [[SHAPE_0:%.+]] = tosa.const_shape {value = dense<[1, 3, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_0]] : (tensor<3x6xf32>, !tosa.shape<3>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[SHAPE_1:%.+]] = tosa.const_shape {value = dense<[1, 6, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_1]] : (tensor<6x4xf32>, !tosa.shape<3>) -> tensor<1x6x4xf32>
// CHECK:           [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<1.618000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           [[ZERO:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>  
// CHECK:           [[VAR_3_:%.+]] = tosa.mul [[VAR_2_]], [[VAR_0_]], [[ZERO]] : (tensor<1x1x1xf32>, tensor<1x3x6xf32>, tensor<1xi8>) -> tensor<1x3x6xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.matmul [[VAR_3_]], [[VAR_1_]] : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[SHAPE_2:%.+]] = tosa.const_shape {value = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_5_:%.+]] = tosa.reshape [[PARAM_2_]], [[SHAPE_2]] : (tensor<3x4xf32>, !tosa.shape<3>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.add [[VAR_4_]], [[VAR_5_]] : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[SHAPE_3:%.+]] = tosa.const_shape {value = dense<[3, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_6_]], [[SHAPE_3]] : (tensor<1x3x4xf32>, !tosa.shape<2>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_7_]] : tensor<3x4xf32>
// CHECK:         }
}

// -----

func.func @test_beta(%arg0: tensor<3x6xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<3x6xf32>) -> tensor<3x6xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {beta = 1.349 : f32} : (tensor<3x6xf32>, tensor<6x6xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
  return %0 : tensor<3x6xf32>
// CHECK-LABEL:  func.func @test_beta
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x6xf32>, [[PARAM_1_:%.+]]: tensor<6x6xf32>, [[PARAM_2_:%.+]]: tensor<3x6xf32>) -> tensor<3x6xf32> {
// CHECK-DAG:       [[SHAPE_0:%.+]] = tosa.const_shape {value = dense<[1, 3, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_0]] : (tensor<3x6xf32>, !tosa.shape<3>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[SHAPE_1:%.+]] = tosa.const_shape {value = dense<[1, 6, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_1]] : (tensor<6x6xf32>, !tosa.shape<3>) -> tensor<1x6x6xf32>
// CHECK:           [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<1.349000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[SHAPE_2:%.+]] = tosa.const_shape {value = dense<[1, 3, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_3_:%.+]] = tosa.reshape [[PARAM_2_]], [[SHAPE_2]] : (tensor<3x6xf32>, !tosa.shape<3>) -> tensor<1x3x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[ZERO:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_4_:%.+]] = tosa.mul [[VAR_2_]], [[VAR_3_]], [[ZERO]] : (tensor<1x1x1xf32>, tensor<1x3x6xf32>, tensor<1xi8>) -> tensor<1x3x6xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.matmul [[VAR_0_]], [[VAR_1_]] : (tensor<1x3x6xf32>, tensor<1x6x6xf32>) -> tensor<1x3x6xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.add [[VAR_5_]], [[VAR_4_]] : (tensor<1x3x6xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
// CHECK-DAG:       [[SHAPE_3:%.+]] = tosa.const_shape {value = dense<[3, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_6_]], [[SHAPE_3]] : (tensor<1x3x6xf32>, !tosa.shape<2>) -> tensor<3x6xf32>
// CHECK:           return [[VAR_7_]] : tensor<3x6xf32>
// CHECK:         }
}

// -----

func.func @test_transa(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> { 
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transA = 1 : si64} : (tensor<6x3xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:  func.func @test_transa
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6x3xf32>, [[PARAM_1_:%.+]]: tensor<6x4xf32>, [[PARAM_2_:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK-DAG:       [[SHAPE_0:%.+]] = tosa.const_shape {value = dense<[1, 6, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_0]] : (tensor<6x3xf32>, !tosa.shape<3>) -> tensor<1x6x3xf32>
// CHECK-DAG:       [[SHAPE_1:%.+]] = tosa.const_shape {value = dense<[1, 6, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_1]] : (tensor<6x4xf32>, !tosa.shape<3>) -> tensor<1x6x4xf32>
// CHECK:           [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_0_]], [[VAR_2_]] : (tensor<1x6x3xf32>, tensor<3xi32>) -> tensor<1x3x6xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.matmul [[VAR_3_]], [[VAR_1_]] : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[SHAPE_2:%.+]] = tosa.const_shape {value = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_5_:%.+]] = tosa.reshape [[PARAM_2_]], [[SHAPE_2]] : (tensor<3x4xf32>, !tosa.shape<3>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.add [[VAR_4_]], [[VAR_5_]] : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-DAG:       [[SHAPE_3:%.+]] = tosa.const_shape {value = dense<[3, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_6_]], [[SHAPE_3]] : (tensor<1x3x4xf32>, !tosa.shape<2>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_7_]] : tensor<3x4xf32>
// CHECK:         }
}

// -----
  
func.func @test_transb(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.184 : f32, transB = 1 : si64} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
// CHECK-LABEL:  func.func @test_transb
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x6xf32>, [[PARAM_1_:%.+]]: tensor<4x6xf32>, [[PARAM_2_:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK:           [[SHAPE_0:%.+]] = tosa.const_shape {value = dense<[1, 3, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_0]] : (tensor<3x6xf32>, !tosa.shape<3>) -> tensor<1x3x6xf32>
// CHECK:           [[SHAPE_1:%.+]] = tosa.const_shape {value = dense<[1, 4, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_1]] : (tensor<4x6xf32>, !tosa.shape<3>) -> tensor<1x4x6xf32>
// CHECK:           [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_2_]] : (tensor<1x4x6xf32>, tensor<3xi32>) -> tensor<1x6x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<1.184000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           [[ZERO:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_5_:%.+]] = tosa.mul [[VAR_4_]], [[VAR_0_]], [[ZERO]] : (tensor<1x1x1xf32>, tensor<1x3x6xf32>, tensor<1xi8>) -> tensor<1x3x6xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.matmul [[VAR_5_]], [[VAR_3_]] : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[SHAPE_2:%.+]] = tosa.const_shape {value = dense<[1, 3, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[PARAM_2_]], [[SHAPE_2]] : (tensor<3x4xf32>, !tosa.shape<3>) -> tensor<1x3x4xf32>
// CHECK:           [[VAR_8_:%.+]] = tosa.add [[VAR_6_]], [[VAR_7_]] : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK:           [[SHAPE_3:%.+]] = tosa.const_shape {value = dense<[3, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_9_:%.+]] = tosa.reshape [[VAR_8_]], [[SHAPE_3]] : (tensor<1x3x4xf32>, !tosa.shape<2>) -> tensor<3x4xf32>
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
// CHECK:           [[SHAPE_0:%.+]] = tosa.const_shape {value = dense<[1, 1, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_0]] : (tensor<1x5xf32>, !tosa.shape<3>) -> tensor<1x1x5xf32>
// CHECK-DAG:       [[SHAPE_1:%.+]] = tosa.const_shape {value = dense<[1, 5, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_2_:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_1]] : (tensor<5x5xf32>, !tosa.shape<3>) -> tensor<1x5x5xf32>
// CHECK:           [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x5x5xf32>, tensor<3xi32>) -> tensor<1x5x5xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.matmul [[VAR_1_]], [[VAR_4_]] : (tensor<1x1x5xf32>, tensor<1x5x5xf32>) -> tensor<1x1x5xf32>
// CHECK-DAG:       [[SHAPE_2:%.+]] = tosa.const_shape {value = dense<[1, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_6_:%.+]] = tosa.reshape [[VAR_5_]], [[SHAPE_2]] : (tensor<1x1x5xf32>, !tosa.shape<2>) -> tensor<1x5xf32>
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
// CHECK-DAG:       [[SHAPE_0:%.+]] = tosa.const_shape {value = dense<[1, 1, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_0]] : (tensor<1x5xf32>, !tosa.shape<3>) -> tensor<1x1x5xf32>
// CHECK-DAG:       [[SHAPE_1:%.+]] = tosa.const_shape {value = dense<[1, 5, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_2_:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_1]] : (tensor<5x6xf32>, !tosa.shape<3>) -> tensor<1x5x6xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<1.349000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[ZERO:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_4_:%.+]] = tosa.mul [[VAR_3_]], [[VAR_1_]], [[ZERO]] : (tensor<1x1x1xf32>, tensor<1x1x5xf32>, tensor<1xi8>) -> tensor<1x1x5xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.matmul [[VAR_4_]], [[VAR_2_]] : (tensor<1x1x5xf32>, tensor<1x5x6xf32>) -> tensor<1x1x6xf32>
// CHECK-DAG:       [[SHAPE_2:%.+]] = tosa.const_shape {value = dense<[1, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_6_:%.+]] = tosa.reshape [[VAR_5_]], [[SHAPE_2]] : (tensor<1x1x6xf32>, !tosa.shape<2>) -> tensor<1x6xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x6xf32>
// CHECK:         }
}

// -----

func.func @test_mixed(%arg0: tensor<11x5xf32>, %arg1: tensor<3x11xf32>, %arg2: tensor<5x3xf32>) -> tensor<5x3xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.402 : f32, beta = 1.998 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<11x5xf32>, tensor<3x11xf32>, tensor<5x3xf32>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
// CHECK-LABEL:  func.func @test_mixed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<11x5xf32>, [[PARAM_1_:%.+]]: tensor<3x11xf32>, [[PARAM_2_:%.+]]: tensor<5x3xf32>) -> tensor<5x3xf32> {
// CHECK-DAG:       [[SHAPE_0:%.+]] = tosa.const_shape {value = dense<[1, 11, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]], [[SHAPE_0]] : (tensor<11x5xf32>, !tosa.shape<3>) -> tensor<1x11x5xf32>
// CHECK:           [[SHAPE_1:%.+]] = tosa.const_shape {value = dense<[1, 3, 11]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[SHAPE_1]] : (tensor<3x11xf32>, !tosa.shape<3>) -> tensor<1x3x11xf32>
// CHECK:           [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_0_]], [[VAR_2_]] : (tensor<1x11x5xf32>, tensor<3xi32>) -> tensor<1x5x11xf32>
// CHECK:           [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_4_]] : (tensor<1x3x11xf32>, tensor<3xi32>) -> tensor<1x11x3xf32>
// CHECK:           [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<1.402000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[ZERO_0:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_7_:%.+]] = tosa.mul [[VAR_6_]], [[VAR_3_]], [[ZERO_0]] : (tensor<1x1x1xf32>, tensor<1x5x11xf32>, tensor<1xi8>) -> tensor<1x5x11xf32>
// CHECK:           [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<1.998000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           [[SHAPE_2:%.+]] = tosa.const_shape {value = dense<[1, 5, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_9_:%.+]] = tosa.reshape [[PARAM_2_]], [[SHAPE_2]] : (tensor<5x3xf32>, !tosa.shape<3>) -> tensor<1x5x3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[ZERO_1:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_10_:%.+]] = tosa.mul [[VAR_8_]], [[VAR_9_]], [[ZERO_1]] : (tensor<1x1x1xf32>, tensor<1x5x3xf32>, tensor<1xi8>) -> tensor<1x5x3xf32>
// CHECK:           [[VAR_11_:%.+]] = tosa.matmul [[VAR_7_]], [[VAR_5_]] : (tensor<1x5x11xf32>, tensor<1x11x3xf32>) -> tensor<1x5x3xf32>
// CHECK:           [[VAR_12_:%.+]] = tosa.add [[VAR_11_]], [[VAR_10_]] : (tensor<1x5x3xf32>, tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
// CHECK:           [[SHAPE_3:%.+]] = tosa.const_shape {value = dense<[5, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_13_:%.+]] = tosa.reshape [[VAR_12_]], [[SHAPE_3]] : (tensor<1x5x3xf32>, !tosa.shape<2>) -> tensor<5x3xf32>
// CHECK:           return [[VAR_13_]] : tensor<5x3xf32>
// CHECK:         }
}

// -----

func.func @gemm(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
// CHECK-LABEL:  func.func @gemm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>, [[PARAM_2_:%.+]]: tensor<4xf32>) -> tensor<1x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<1x5xf32>, !tosa.shape<3>) -> tensor<1x1x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_2_]] : (tensor<4x5xf32>, !tosa.shape<3>) -> tensor<1x4x5xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.transpose [[VAR_3_]], [[VAR_4_]] : (tensor<1x4x5xf32>, tensor<3xi32>) -> tensor<1x5x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.matmul [[VAR_1_]], [[VAR_5_]] : (tensor<1x1x5xf32>, tensor<1x5x4xf32>) -> tensor<1x1x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.reshape [[PARAM_2_]], [[VAR_7_]] : (tensor<4xf32>, !tosa.shape<3>) -> tensor<1x1x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.add [[VAR_6_]], [[VAR_8_]] : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.const_shape  {value = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_11_:%.+]] = tosa.reshape [[VAR_9_]], [[VAR_10_]] : (tensor<1x1x4xf32>, !tosa.shape<2>) -> tensor<1x4xf32>
// CHECK:           return [[VAR_11_]] : tensor<1x4xf32>
// CHECK:         }
}
  
// -----
  
func.func @gemm_broadcast(%arg0: tensor<2x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<2x5xf32>, tensor<4x5xf32>, tensor<1xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
// CHECK-LABEL:  func.func @gemm_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>, [[PARAM_2_:%.+]]: tensor<1xf32>) -> tensor<2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 2, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<2x5xf32>, !tosa.shape<3>) -> tensor<1x2x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_2_]] : (tensor<4x5xf32>, !tosa.shape<3>) -> tensor<1x4x5xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.transpose [[VAR_3_]], [[VAR_4_]] : (tensor<1x4x5xf32>, tensor<3xi32>) -> tensor<1x5x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.matmul [[VAR_1_]], [[VAR_5_]] : (tensor<1x2x5xf32>, tensor<1x5x4xf32>) -> tensor<1x2x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.const_shape  {value = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.reshape [[PARAM_2_]], [[VAR_7_]] : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.add [[VAR_6_]], [[VAR_8_]] : (tensor<1x2x4xf32>, tensor<1x1x1xf32>) -> tensor<1x2x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.const_shape  {value = dense<[2, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_11_:%.+]] = tosa.reshape [[VAR_9_]], [[VAR_10_]] : (tensor<1x2x4xf32>, !tosa.shape<2>) -> tensor<2x4xf32>
// CHECK:           return [[VAR_11_]] : tensor<2x4xf32>
// CHECK:         }
}

// -----

func.func @gemm_no_bias(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, none) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
// CHECK-LABEL:  func.func @gemm_no_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>) -> tensor<1x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_1_]] : (tensor<1x5xf32>, !tosa.shape<3>) -> tensor<1x1x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[1, 4, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_3_]] : (tensor<4x5xf32>, !tosa.shape<3>) -> tensor<1x4x5xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.transpose [[VAR_4_]], [[VAR_5_]] : (tensor<1x4x5xf32>, tensor<3xi32>) -> tensor<1x5x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.matmul [[VAR_2_]], [[VAR_6_]] : (tensor<1x1x5xf32>, tensor<1x5x4xf32>) -> tensor<1x1x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_9_:%.+]] = tosa.reshape [[VAR_7_]], [[VAR_8_]] : (tensor<1x1x4xf32>, !tosa.shape<2>) -> tensor<1x4xf32>
// CHECK:           return [[VAR_9_]] : tensor<1x4xf32>
// CHECK:         }

}
