// RUN: onnx-mlir-opt --canonicalize --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

// COM: Test rewriting GlobalAveragePool into ReduceMean
func.func @test_global_average_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  return %0 : tensor<1x3x1x1xf32>
}

// CHECK-LABEL:  func.func @test_global_average_pool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [1, 3, 1, 1] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<2.500000e+01> : tensor<1x3x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_2_]]) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x3x5x5xf32>, tensor<f32>) -> tensor<1x3xf32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.dynamic_reshape [[VAR_3_]], [[VAR_0_]] : (tensor<1x3xf32>, tensor<4xindex>) -> tensor<1x3x1x1xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.divide [[VAR_4_]], [[VAR_1_]] : tensor<1x3x1x1xf32>
// CHECK:           return [[VAR_5_]] : tensor<1x3x1x1xf32>
// CHECK:         }

// -----

// COM: Test rewriting GlobalAveragePool into ReduceMean with dynamic dimensions
func.func @test_global_average_pool_dyn_dims(%arg0: tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32>
  return %0 : tensor<1x?x1x1xf32>
}

// CHECK-LABEL:  func.func @test_global_average_pool_dyn_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x?x?x5xf32>, tensor<f32>) -> tensor<1x?xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<1x?x?x5xf32> -> tensor<4xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.get_extent [[VAR_3_]], [[CST_0_]] : tensor<4xindex>, index -> index
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.get_extent [[VAR_3_]], [[CST_1_]] : tensor<4xindex>, index -> index
// CHECK:           [[VAR_6_:%.+]] = shape.from_extents [[VAR_4_]], [[VAR_5_]], [[CST_1_]], [[CST_1_]] : index, index, index, index
// CHECK:           [[VAR_7_:%.+]] = shape.to_extent_tensor [[VAR_6_]] : !shape.shape -> tensor<4xindex>
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.dynamic_reshape [[VAR_2_]], [[VAR_7_]] : (tensor<1x?xf32>, tensor<4xindex>) -> tensor<1x?x1x1xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<1x?x?x5xf32> -> tensor<4xindex>
// CHECK:           [[VAR_10_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_0_]], [[VAR_9_]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<1x?x?x5xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.reduce([[VAR_10_]] init: [[VAR_1_]]) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x?x?x5xf32>, tensor<f32>) -> tensor<1x?xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = shape.shape_of [[VAR_10_]] : tensor<1x?x?x5xf32> -> tensor<4xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = shape.get_extent [[VAR_12_]], [[CST_0_]] : tensor<4xindex>, index -> index
// CHECK-DAG:       [[VAR_14_:%.+]] = shape.get_extent [[VAR_12_]], [[CST_1_]] : tensor<4xindex>, index -> index
// CHECK:           [[VAR_15_:%.+]] = shape.from_extents [[VAR_13_]], [[VAR_14_]], [[CST_1_]], [[CST_1_]] : index, index, index, index
// CHECK:           [[VAR_16_:%.+]] = shape.to_extent_tensor [[VAR_15_]] : !shape.shape -> tensor<4xindex>
// CHECK:           [[VAR_17_:%.+]] = stablehlo.dynamic_reshape [[VAR_11_]], [[VAR_16_]] : (tensor<1x?xf32>, tensor<4xindex>) -> tensor<1x?x1x1xf32>
// CHECK:           [[VAR_18_:%.+]] = stablehlo.divide [[VAR_8_]], [[VAR_17_]] : tensor<1x?x1x1xf32>
// CHECK:           return [[VAR_18_]] : tensor<1x?x1x1xf32>
// CHECK:         }

// -----

// COM: Test rewriting GlobalMaxPool into ReduceMax
func.func @test_global_max_pool(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
  return %0 : tensor<1x3x1x1xf32>
}

// CHECK-LABEL:  func.func @test_global_max_pool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [1, 3, 1, 1] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies stablehlo.maximum across dimensions = [2, 3] : (tensor<1x3x5x5xf32>, tensor<f32>) -> tensor<1x3xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.dynamic_reshape [[VAR_2_]], [[VAR_0_]] : (tensor<1x3xf32>, tensor<4xindex>) -> tensor<1x3x1x1xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x3x1x1xf32>
// CHECK:         }

// -----

// COM: Test rewriting GlobalMaxPool into ReduceMax with dynamic dimensions
func.func @test_global_max_pool_dyn_dims(%arg0: tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32>
  return %0 : tensor<1x?x1x1xf32>
}

// CHECK-LABEL:  func.func @test_global_max_pool_dyn_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x5xf32>) -> tensor<1x?x1x1xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_0_]]) applies stablehlo.maximum across dimensions = [2, 3] : (tensor<1x?x?x5xf32>, tensor<f32>) -> tensor<1x?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<1x?x?x5xf32> -> tensor<4xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.get_extent [[VAR_2_]], [[CST_0_]] : tensor<4xindex>, index -> index
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.get_extent [[VAR_2_]], [[CST_1_]] : tensor<4xindex>, index -> index
// CHECK:           [[VAR_5_:%.+]] = shape.from_extents [[VAR_3_]], [[VAR_4_]], [[CST_1_]], [[CST_1_]] : index, index, index, index
// CHECK:           [[VAR_6_:%.+]] = shape.to_extent_tensor [[VAR_5_]] : !shape.shape -> tensor<4xindex>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.dynamic_reshape [[VAR_1_]], [[VAR_6_]] : (tensor<1x?xf32>, tensor<4xindex>) -> tensor<1x?x1x1xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x?x1x1xf32>
// CHECK:         }
