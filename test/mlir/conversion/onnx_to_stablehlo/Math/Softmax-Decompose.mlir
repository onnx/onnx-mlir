// RUN: onnx-mlir-opt --decompose-onnx="target=stablehlo" --convert-onnx-to-stablehlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_softmax(%arg0 : tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64} : (tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
  "func.return"(%0) : (tensor<10x20x30xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_softmax
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [10, 1, 30] : tensor<3xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.const_shape [10, 20, 30] : tensor<3xindex>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_3_]]) applies stablehlo.maximum across dimensions = [1] : (tensor<10x20x30xf32>, tensor<f32>) -> tensor<10x30xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.dynamic_reshape [[VAR_4_]], [[VAR_0_]] : (tensor<10x30xf32>, tensor<3xindex>) -> tensor<10x1x30xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_1_]], dims = [0, 1, 2] : (tensor<10x20x30xf32>, tensor<3xindex>) -> tensor<10x20x30xf32>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_5_]], [[VAR_1_]], dims = [0, 1, 2] : (tensor<10x1x30xf32>, tensor<3xindex>) -> tensor<10x20x30xf32>
// CHECK:           [[VAR_8_:%.+]] = stablehlo.subtract [[VAR_6_]], [[VAR_7_]] : tensor<10x20x30xf32>
// CHECK:           [[VAR_9_:%.+]] = stablehlo.exponential [[VAR_8_]] : tensor<10x20x30xf32>
// CHECK:           [[VAR_10_:%.+]] = stablehlo.reduce([[VAR_9_]] init: [[VAR_2_]]) applies stablehlo.add across dimensions = [1] : (tensor<10x20x30xf32>, tensor<f32>) -> tensor<10x30xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.dynamic_reshape [[VAR_10_]], [[VAR_0_]] : (tensor<10x30xf32>, tensor<3xindex>) -> tensor<10x1x30xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_9_]], [[VAR_1_]], dims = [0, 1, 2] : (tensor<10x20x30xf32>, tensor<3xindex>) -> tensor<10x20x30xf32>
// CHECK:           [[VAR_13_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_11_]], [[VAR_1_]], dims = [0, 1, 2] : (tensor<10x1x30xf32>, tensor<3xindex>) -> tensor<10x20x30xf32>
// CHECK:           [[VAR_14_:%.+]] = stablehlo.divide [[VAR_12_]], [[VAR_13_]] : tensor<10x20x30xf32>
// CHECK:           return [[VAR_14_]] : tensor<10x20x30xf32>
// CHECK:         }

// -----

func.func @test_softmax_dynamic(%arg0 : tensor<?x20x30xf32>) -> tensor<?x20x30xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64} : (tensor<?x20x30xf32>) -> tensor<?x20x30xf32>
  "func.return"(%0) : (tensor<?x20x30xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_softmax_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20x30xf32>) -> tensor<?x20x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies stablehlo.maximum across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = shape.get_extent [[VAR_3_]], [[CST_0_]] : tensor<3xindex>, index -> index
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.get_extent [[VAR_3_]], [[CST_2_]] : tensor<3xindex>, index -> index
// CHECK:           [[VAR_6_:%.+]] = shape.from_extents [[VAR_4_]], [[CST_1_]], [[VAR_5_]] : index, index, index
// CHECK:           [[VAR_7_:%.+]] = shape.to_extent_tensor [[VAR_6_]] : !shape.shape -> tensor<3xindex>
// CHECK-DAG:       [[VAR_8_:%.+]] = stablehlo.dynamic_reshape [[VAR_2_]], [[VAR_7_]] : (tensor<?x30xf32>, tensor<3xindex>) -> tensor<?x1x30xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK:           [[VAR_10_:%.+]] = shape.shape_of [[VAR_8_]] : tensor<?x1x30xf32> -> tensor<3xindex>
// CHECK:           [[VAR_11_:%.+]] = shape.broadcast [[VAR_9_]], [[VAR_10_]] : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_11_]], dims = [0, 1, 2] : (tensor<?x20x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_8_]], [[VAR_11_]], dims = [0, 1, 2] : (tensor<?x1x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK:           [[VAR_14_:%.+]] = stablehlo.subtract [[VAR_12_]], [[VAR_13_]] : tensor<?x20x30xf32>
// CHECK:           [[VAR_15_:%.+]] = stablehlo.exponential [[VAR_14_]] : tensor<?x20x30xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = stablehlo.reduce([[VAR_15_]] init: [[VAR_0_]]) applies stablehlo.add across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = shape.shape_of [[VAR_15_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = shape.get_extent [[VAR_17_]], [[CST_0_]] : tensor<3xindex>, index -> index
// CHECK-DAG:       [[VAR_19_:%.+]] = shape.get_extent [[VAR_17_]], [[CST_2_]] : tensor<3xindex>, index -> index
// CHECK:           [[VAR_20_:%.+]] = shape.from_extents [[VAR_18_]], [[CST_1_]], [[VAR_19_]] : index, index, index
// CHECK:           [[VAR_21_:%.+]] = shape.to_extent_tensor [[VAR_20_]] : !shape.shape -> tensor<3xindex>
// CHECK-DAG:       [[VAR_22_:%.+]] = stablehlo.dynamic_reshape [[VAR_16_]], [[VAR_21_]] : (tensor<?x30xf32>, tensor<3xindex>) -> tensor<?x1x30xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = shape.shape_of [[VAR_15_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK:           [[VAR_24_:%.+]] = shape.shape_of [[VAR_22_]] : tensor<?x1x30xf32> -> tensor<3xindex>
// CHECK:           [[VAR_25_:%.+]] = shape.broadcast [[VAR_23_]], [[VAR_24_]] : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-DAG:       [[VAR_26_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_15_]], [[VAR_25_]], dims = [0, 1, 2] : (tensor<?x20x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_22_]], [[VAR_25_]], dims = [0, 1, 2] : (tensor<?x1x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK:           [[VAR_28_:%.+]] = stablehlo.divide [[VAR_26_]], [[VAR_27_]] : tensor<?x20x30xf32>
// CHECK:           return [[VAR_28_]] : tensor<?x20x30xf32>
// CHECK:         }


// -----

func.func @test_softmax_2d(%arg0 : tensor<1x10xf32>) -> tensor<1x10xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64} : (tensor<1x10xf32>) -> tensor<1x10xf32>
  "func.return"(%0) : (tensor<1x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_softmax_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x10xf32>) -> tensor<1x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [1, 1] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.const_shape [1, 10] : tensor<2xindex>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.reduce([[PARAM_0_]] init: [[VAR_3_]]) applies stablehlo.maximum across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.dynamic_reshape [[VAR_4_]], [[VAR_0_]] : (tensor<1xf32>, tensor<2xindex>) -> tensor<1x1xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_1_]], dims = [0, 1] : (tensor<1x10xf32>, tensor<2xindex>) -> tensor<1x10xf32>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_5_]], [[VAR_1_]], dims = [0, 1] : (tensor<1x1xf32>, tensor<2xindex>) -> tensor<1x10xf32>
// CHECK:           [[VAR_8_:%.+]] = stablehlo.subtract [[VAR_6_]], [[VAR_7_]] : tensor<1x10xf32>
// CHECK:           [[VAR_9_:%.+]] = stablehlo.exponential [[VAR_8_]] : tensor<1x10xf32>
// CHECK:           [[VAR_10_:%.+]] = stablehlo.reduce([[VAR_9_]] init: [[VAR_2_]]) applies stablehlo.add across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = stablehlo.dynamic_reshape [[VAR_10_]], [[VAR_0_]] : (tensor<1xf32>, tensor<2xindex>) -> tensor<1x1xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_9_]], [[VAR_1_]], dims = [0, 1] : (tensor<1x10xf32>, tensor<2xindex>) -> tensor<1x10xf32>
// CHECK:           [[VAR_13_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_11_]], [[VAR_1_]], dims = [0, 1] : (tensor<1x1xf32>, tensor<2xindex>) -> tensor<1x10xf32>
// CHECK:           [[VAR_14_:%.+]] = stablehlo.divide [[VAR_12_]], [[VAR_13_]] : tensor<1x10xf32>
// CHECK:           return [[VAR_14_]] : tensor<1x10xf32>
// CHECK:         }
