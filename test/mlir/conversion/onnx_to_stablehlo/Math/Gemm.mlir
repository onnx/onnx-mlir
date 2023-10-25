// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo %s -split-input-file | FileCheck %s

func.func @test_gemm_bias_none(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<10x10xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none
  %0 ="onnx.Gemm"(%arg0, %arg1, %bias) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, none) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_gemm_bias_none
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.dot [[PARAM_0_]], [[PARAM_1_]] : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_gemm_bias_1d(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<10x10xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_gemm_bias_1d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.dot [[PARAM_0_]], [[PARAM_1_]] : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_2_]], dims = [1] : (tensor<10xf32>) -> tensor<10x10xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.add [[VAR_0_]], [[VAR_1_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_2_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_gemm_bias_2d(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 2.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_gemm_bias_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<10x10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dot [[PARAM_0_]], [[PARAM_1_]] : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.multiply [[VAR_1_]], [[VAR_0_]] : tensor<10x10xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.add [[VAR_2_]], [[PARAM_2_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_gemm_transA(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<10x10xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 2.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_gemm_transA
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x10xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<10x10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.transpose [[PARAM_0_]], dims = [1, 0] : (tensor<5x10xf32>) -> tensor<10x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.dot [[VAR_1_]], [[PARAM_1_]] : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_2_]], dims = [1] : (tensor<10xf32>) -> tensor<10x10xf32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.multiply [[VAR_3_]], [[VAR_0_]] : tensor<10x10xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.add [[VAR_2_]], [[VAR_4_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_5_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_gemm_transAB(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<5xf32>) -> tensor<5x5xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 2.0 : f32, beta = 3.0 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<10x5xf32>, tensor<5x10xf32>, tensor<5xf32>) -> tensor<5x5xf32>
 "func.return"(%0) : (tensor<5x5xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_gemm_transAB
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<5xf32>) -> tensor<5x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.transpose [[PARAM_0_]], dims = [1, 0] : (tensor<10x5xf32>) -> tensor<5x10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.transpose [[PARAM_1_]], dims = [1, 0] : (tensor<5x10xf32>) -> tensor<10x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.dot [[VAR_0_]], [[VAR_1_]] : (tensor<5x10xf32>, tensor<10x5xf32>) -> tensor<5x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<5x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.multiply [[VAR_2_]], [[VAR_3_]] : tensor<5x5xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_2_]], dims = [1] : (tensor<5xf32>) -> tensor<5x5xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.constant dense<3.000000e+00> : tensor<5x5xf32>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.multiply [[VAR_5_]], [[VAR_6_]] : tensor<5x5xf32>
// CHECK:           [[VAR_8_:%.+]] = stablehlo.add [[VAR_4_]], [[VAR_7_]] : tensor<5x5xf32>
// CHECK:           return [[VAR_8_]] : tensor<5x5xf32>
// CHECK:         }

// -----

func.func @test_gemm_unknown_dims(%arg0: tensor<?x5xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<?x10xf32> {
  %0= "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<?x5xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_gemm_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.dot [[PARAM_0_]], [[PARAM_1_]] : (tensor<?x5xf32>, tensor<5x10xf32>) -> tensor<?x10xf32>
// CHECK:           [[VAR_1_:%.+]] = shape.shape_of [[VAR_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_2_]], [[VAR_1_]], dims = [1] : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.add [[VAR_0_]], [[VAR_2_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_gemm_unknown_dims_not_lowered(%arg0: tensor<?x5xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<?xf32>) -> tensor<?x10xf32> {
  %0= "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<?x5xf32>, tensor<5x10xf32>, tensor<?xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_gemm_unknown_dims_not_lowered
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x5xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<?xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.dot [[PARAM_0_]], [[PARAM_1_]] : (tensor<?x5xf32>, tensor<5x10xf32>) -> tensor<?x10xf32>
// CHECK:           [[VAR_1_:%.+]] = shape.shape_of [[VAR_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_2_]], [[VAR_1_]], dims = [1] : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.add [[VAR_0_]], [[VAR_2_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_gemm_not_lowered(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<10x10xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 2.0 : f32, beta = 1.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
 "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_gemm_not_lowered
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x10xf32>, [[PARAM_1_:%.+]]: tensor<5x10xf32>, [[PARAM_2_:%.+]]: tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.transpose [[PARAM_0_]], dims = [1, 0] : (tensor<5x10xf32>) -> tensor<10x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dot [[VAR_0_]], [[PARAM_1_]] : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<10x10xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<2.000000e+00> : tensor<10x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.multiply [[VAR_1_]], [[VAR_2_]] : tensor<10x10xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_2_]], dims = [1] : (tensor<10xf32>) -> tensor<10x10xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.add [[VAR_3_]], [[VAR_4_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_5_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_exceed_limit_gemm(%arg0 : tensor<32769x5xf32>, %arg1 : tensor<5x32769xf32>, %arg2: tensor<32769xf32>) -> tensor<32769x32769xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<32769x5xf32>, tensor<5x32769xf32>, tensor<32769xf32>) -> tensor<32769x32769xf32>
 "func.return"(%0) : (tensor<32769x32769xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_exceed_limit_gemm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<32769x5xf32>, [[PARAM_1_:%.+]]: tensor<5x32769xf32>, [[PARAM_2_:%.+]]: tensor<32769xf32>) -> tensor<32769x32769xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.dot [[PARAM_0_]], [[PARAM_1_]] : (tensor<32769x5xf32>, tensor<5x32769xf32>) -> tensor<32769x32769xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.broadcast_in_dim [[PARAM_2_]], dims = [1] : (tensor<32769xf32>) -> tensor<32769x32769xf32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.add [[VAR_0_]], [[VAR_1_]] : tensor<32769x32769xf32>
// CHECK:           return [[VAR_2_]] : tensor<32769x32769xf32>
// CHECK:         }
