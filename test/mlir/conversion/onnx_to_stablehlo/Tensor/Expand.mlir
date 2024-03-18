// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

// -----

func.func @test_expand_with_arith_constant(%arg0 : tensor<2x1x6x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[7, 1, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_expand_with_arith_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x6x1xf32>) -> tensor<2x7x6x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 7, 6, 5] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.broadcast_in_dim [[VAR_1_]], dims = [] : (tensor<f32>) -> tensor<7x1x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_0_]], dims = [0, 1, 2, 3] : (tensor<2x1x6x1xf32>, tensor<4xindex>) -> tensor<2x7x6x5xf32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_2_]], [[VAR_0_]], dims = [1, 2, 3] : (tensor<7x1x5xf32>, tensor<4xindex>) -> tensor<2x7x6x5xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.multiply [[VAR_3_]], [[VAR_4_]] : tensor<2x7x6x5xf32>
// CHECK:           return [[VAR_5_]] : tensor<2x7x6x5xf32>
// CHECK:         }

// -----

func.func @test_expand_integer_tensor(%arg0 : tensor<2x1x6x1xi64>) -> tensor<*xi64> {
  %0 = "onnx.Constant"() {value = dense<[7, 1, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xi64>, tensor<3xi64>) -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
}

// CHECK-LABEL:  func.func @test_expand_integer_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x6x1xi64>) -> tensor<2x7x6x5xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 7, 6, 5] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.broadcast_in_dim [[VAR_1_]], dims = [] : (tensor<i64>) -> tensor<7x1x5xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_0_]], dims = [0, 1, 2, 3] : (tensor<2x1x6x1xi64>, tensor<4xindex>) -> tensor<2x7x6x5xi64>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_2_]], [[VAR_0_]], dims = [1, 2, 3] : (tensor<7x1x5xi64>, tensor<4xindex>) -> tensor<2x7x6x5xi64>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.multiply [[VAR_3_]], [[VAR_4_]] : tensor<2x7x6x5xi64>
// CHECK:           return [[VAR_5_]] : tensor<2x7x6x5xi64>
// CHECK:         }

// -----

func.func @test_expand_with_shape(%arg0 : tensor<2x1x6x1xf32>, %arg1: tensor<6x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Shape"(%arg1) : (tensor<6x2xf32>) -> tensor<*xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<*xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_expand_with_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x6x1xf32>, [[PARAM_1_:%.+]]: tensor<6x2xf32>) -> tensor<2x1x6x2xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<[6, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 1, 6, 1] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_1_]], [[VAR_cst_]], dims = [] : (tensor<f32>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:           [[VAR_3_:%.+]] = shape.shape_of [[VAR_2_]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           [[VAR_4_:%.+]] = shape.broadcast [[VAR_3_]], [[VAR_0_]] : tensor<2xindex>, tensor<4xindex> -> tensor<4xindex>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_4_]], dims = [0, 1, 2, 3] : (tensor<2x1x6x1xf32>, tensor<4xindex>) -> tensor<2x1x6x2xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_2_]], [[VAR_4_]], dims = [2, 3] : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<2x1x6x2xf32>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.multiply [[VAR_5_]], [[VAR_6_]] : tensor<2x1x6x2xf32>
// CHECK:           return [[VAR_7_]] : tensor<2x1x6x2xf32>
// CHECK:         }

// -----

  func.func @test_expand_with_arbitrary(%arg0: tensor<2x1x6x1xf32>, %arg1: tensor<2xi64>) -> tensor<2x1x6x2xf32> {
    %1 = "onnx.Expand"(%arg0, %arg1) : (tensor<2x1x6x1xf32>, tensor<2xi64>) -> tensor<2x1x6x2xf32>
    return %1 : tensor<2x1x6x2xf32>
  }

// CHECK-LABEL:  func.func @test_expand_with_arbitrary
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x6x1xf32>, [[PARAM_1_:%.+]]: tensor<2xi64>) -> tensor<2x1x6x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [2, 1, 6, 1] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_1_]], [[PARAM_1_]], dims = [] : (tensor<f32>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:           [[VAR_3_:%.+]] = shape.shape_of [[VAR_2_]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           [[VAR_4_:%.+]] = shape.broadcast [[VAR_3_]], [[VAR_0_]] : tensor<2xindex>, tensor<4xindex> -> tensor<4xindex>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_4_]], dims = [0, 1, 2, 3] : (tensor<2x1x6x1xf32>, tensor<4xindex>) -> tensor<2x1x6x2xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_2_]], [[VAR_4_]], dims = [2, 3] : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<2x1x6x2xf32>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.multiply [[VAR_5_]], [[VAR_6_]] : tensor<2x1x6x2xf32>
// CHECK:           return [[VAR_7_]] : tensor<2x1x6x2xf32>
// CHECK:         }
