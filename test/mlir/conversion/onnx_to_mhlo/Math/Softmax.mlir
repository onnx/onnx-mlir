// RUN: onnx-mlir-opt --decompose-onnx="target=mhlo" --convert-onnx-to-mhlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_softmax(%arg0 : tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64} : (tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
  "func.return"(%0) : (tensor<10x20x30xf32>) -> ()
// CHECK-LABEL:  func @test_softmax
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
// CHECK-DAG:     [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     [[VAR_1_:%.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:    [[VAR_2_:%.+]] = mhlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies mhlo.maximum across dimensions = [1] : (tensor<10x20x30xf32>, tensor<f32>) -> tensor<10x30xf32>
// CHECK-NEXT:    [[VAR_3_:%.+]] = mhlo.reshape [[VAR_2_]] : (tensor<10x30xf32>) -> tensor<10x1x30xf32>
// CHECK-NEXT:    [[VAR_4_:%.+]] = "mhlo.broadcast_in_dim"([[VAR_3_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<10x1x30xf32>) -> tensor<10x20x30xf32>
// CHECK-NEXT:    [[VAR_5_:%.+]] = mhlo.subtract [[PARAM_0_]], [[VAR_4_]] : tensor<10x20x30xf32>
// CHECK-NEXT:    [[VAR_6_:%.+]] = mhlo.exponential [[VAR_5_]] : tensor<10x20x30xf32>
// CHECK-NEXT:    [[VAR_7_:%.+]] = mhlo.reduce([[VAR_6_]] init: [[VAR_0_]]) applies mhlo.add across dimensions = [1] : (tensor<10x20x30xf32>, tensor<f32>) -> tensor<10x30xf32>
// CHECK-NEXT:    [[VAR_8_:%.+]] = mhlo.reshape [[VAR_7_]] : (tensor<10x30xf32>) -> tensor<10x1x30xf32>
// CHECK-NEXT:    [[VAR_9_:%.+]] = "mhlo.broadcast_in_dim"([[VAR_8_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<10x1x30xf32>) -> tensor<10x20x30xf32>
// CHECK-NEXT:    [[VAR_10_:%.+]] = mhlo.divide [[VAR_6_]], [[VAR_9_]] : tensor<10x20x30xf32>
// CHECK-NEXT:    return [[VAR_10_]] : tensor<10x20x30xf32>
}

// -----

func.func @test_softmax_dynamic(%arg0 : tensor<?x20x30xf32>) -> tensor<?x20x30xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64} : (tensor<?x20x30xf32>) -> tensor<?x20x30xf32>
  "func.return"(%0) : (tensor<?x20x30xf32>) -> ()
// CHECK-LABEL:  func @test_softmax_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20x30xf32>) -> tensor<?x20x30xf32> {
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     [[VAR_1_:%.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:    [[VAR_2_:%.+]] = mhlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies mhlo.maximum across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-DAG:     [[VAR_3_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-DAG:     [[VAR_4_:%.+]] = shape.get_extent [[VAR_3_]], %c0 : tensor<3xindex>, index -> index
// CHECK-DAG:     [[VAR_5_:%.+]] = shape.get_extent [[VAR_3_]], %c2 : tensor<3xindex>, index -> index
// CHECK-NEXT:    [[VAR_6_:%.+]] = shape.from_extents [[VAR_4_]], %c1, [[VAR_5_]] : index, index, index
// CHECK-NEXT:    [[VAR_7_:%.+]] = shape.to_extent_tensor [[VAR_6_]] : !shape.shape -> tensor<3xindex>
// CHECK-DAG:     [[VAR_8_:%.+]] = mhlo.dynamic_reshape [[VAR_2_]], [[VAR_7_]] : (tensor<?x30xf32>, tensor<3xindex>) -> tensor<?x1x30xf32>
// CHECK-DAG:     [[VAR_9_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-NEXT:    [[VAR_10_:%.+]] = shape.broadcast [[VAR_9_]], [[VAR_7_]] : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-DAG:     [[VAR_11_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[PARAM_0_]], [[VAR_10_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x20x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-DAG:     [[VAR_12_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_8_]], [[VAR_10_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x1x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-NEXT:    [[VAR_13_:%.+]] = mhlo.subtract [[VAR_11_]], [[VAR_12_]] : tensor<?x20x30xf32>
// CHECK-NEXT:    [[VAR_14_:%.+]] = mhlo.exponential [[VAR_13_]] : tensor<?x20x30xf32>
// CHECK-NEXT:    [[VAR_15_:%.+]] = mhlo.reduce([[VAR_14_]] init: [[VAR_0_]]) applies mhlo.add across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-NEXT:    [[VAR_16_:%.+]] = shape.shape_of [[VAR_14_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-DAG:     [[VAR_17_:%.+]] = shape.get_extent [[VAR_16_]], %c0 : tensor<3xindex>, index -> index
// CHECK-DAG:     [[VAR_18_:%.+]] = shape.get_extent [[VAR_16_]], %c2 : tensor<3xindex>, index -> index
// CHECK-NEXT:    [[VAR_19_:%.+]] = shape.from_extents [[VAR_17_]], %c1, [[VAR_18_]] : index, index, index
// CHECK-NEXT:    [[VAR_20_:%.+]] = shape.to_extent_tensor [[VAR_19_]] : !shape.shape -> tensor<3xindex>
// CHECK-NEXT:    [[VAR_21_:%.+]] = mhlo.dynamic_reshape [[VAR_15_]], [[VAR_20_]] : (tensor<?x30xf32>, tensor<3xindex>) -> tensor<?x1x30xf32>
// CHECK-NEXT:    [[VAR_22_:%.+]] = shape.shape_of [[VAR_14_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-NEXT:    [[VAR_23_:%.+]] = shape.broadcast [[VAR_22_]], [[VAR_20_]] : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-DAG:     [[VAR_24_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_14_]], [[VAR_23_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x20x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-DAG:     [[VAR_25_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_21_]], [[VAR_23_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x1x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-NEXT:    [[VAR_26_:%.+]] = mhlo.divide [[VAR_24_]], [[VAR_25_]] : tensor<?x20x30xf32>
// CHECK-NEXT:    return [[VAR_26_]] : tensor<?x20x30xf32>
}

// -----

func.func @test_softmax_2d(%arg0 : tensor<1x10xf32>) -> tensor<1x10xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64} : (tensor<1x10xf32>) -> tensor<1x10xf32>
  "func.return"(%0) : (tensor<1x10xf32>) -> ()
// CHECK-LABEL:  func @test_softmax_2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x10xf32>) -> tensor<1x10xf32> {
// CHECK: [[VAR_2_:%.+]] = mhlo.reduce([[PARAM_0_]] init: [[VAR_1_:%.+]]) applies mhlo.maximum across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK: [[VAR_3_:%.+]] = mhlo.reshape [[VAR_2_]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK: [[VAR_4_:%.+]] = "mhlo.broadcast_in_dim"([[VAR_3_]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x10xf32>
// CHECK: [[VAR_7_:%.+]] = mhlo.reduce([[VAR_6_]] init: [[VAR_0_:%.+]]) applies mhlo.add across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK: [[VAR_8_:%.+]] = mhlo.reshape [[VAR_7_]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK: [[VAR_9_:%.+]] = "mhlo.broadcast_in_dim"([[VAR_8_]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x10xf32>
// CHECK: [[VAR_10_:%.+]] = mhlo.divide [[VAR_6_]], [[VAR_9_]] : tensor<1x10xf32>
}
