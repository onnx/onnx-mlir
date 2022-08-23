// RUN: onnx-mlir-opt --decompose-onnx="target=mhlo" --convert-onnx-to-mhlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_softmax(%arg0 : tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64, onnx_opset = 13 : si64} : (tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
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


func.func @test_softmax_dynamic(%arg0 : tensor<?x20x30xf32>) -> tensor<?x20x30xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64, onnx_opset = 13 : si64} : (tensor<?x20x30xf32>) -> tensor<?x20x30xf32>
  "func.return"(%0) : (tensor<?x20x30xf32>) -> ()
// CHECK-LABEL:  func @test_softmax_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20x30xf32>) -> tensor<?x20x30xf32> {
// CHECK-DAG:     [[VAR_1_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     [[VAR_2_:%.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:    [[VAR_3_:%.+]] = mhlo.reduce([[PARAM_0_]] init: [[VAR_2_]]) applies mhlo.maximum across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-NEXT:    [[VAR_4_:%.+]] = mhlo.dynamic_reshape [[VAR_3_]], [[VAR_0_:%.+]] : (tensor<?x30xf32>, tensor<3xi64>) -> tensor<?x1x30xf32>
// CHECK-DAG:     [[VAR_5_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-DAG:     [[VAR_6_:%.+]] = shape.shape_of [[VAR_4_]] : tensor<?x1x30xf32> -> tensor<3xindex>
// CHECK-NEXT:    [[VAR_7_:%.+]] = shape.broadcast [[VAR_5_]], [[VAR_6_]] : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-DAG:     [[VAR_8_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[PARAM_0_]], [[VAR_7_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x20x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-DAG:     [[VAR_9_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_4_]], [[VAR_7_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x1x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-NEXT:    [[VAR_10_:%.+]] = mhlo.subtract [[VAR_8_]], [[VAR_9_]] : tensor<?x20x30xf32>
// CHECK-NEXT:    [[VAR_11_:%.+]] = mhlo.exponential [[VAR_10_]] : tensor<?x20x30xf32>
// CHECK-NEXT:    [[VAR_12_:%.+]] = mhlo.reduce([[VAR_11_]] init: [[VAR_1_]]) applies mhlo.add across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-NEXT:    [[VAR_13_:%.+]] = mhlo.dynamic_reshape [[VAR_12_]], [[VAR_0_]] : (tensor<?x30xf32>, tensor<3xi64>) -> tensor<?x1x30xf32>
// CHECK-DAG:     [[VAR_14_:%.+]] = shape.shape_of [[VAR_11_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-DAG:     [[VAR_15_:%.+]] = shape.shape_of [[VAR_13_]] : tensor<?x1x30xf32> -> tensor<3xindex>
// CHECK-NEXT:    [[VAR_16_:%.+]] = shape.broadcast [[VAR_14_]], [[VAR_15_]] : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-DAG:     [[VAR_17_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_11_]], [[VAR_16_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x20x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-DAG:     [[VAR_18_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_13_]], [[VAR_16_]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x1x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-NEXT:    [[VAR_19_:%.+]] = mhlo.divide [[VAR_17_]], [[VAR_18_]] : tensor<?x20x30xf32>
// CHECK-NEXT:    return [[VAR_19_]] : tensor<?x20x30xf32>
}

func.func @test_softmax_2d(%arg0 : tensor<1x10xf32>) -> tensor<1x10xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64, onnx_opset = 13 : si64} : (tensor<1x10xf32>) -> tensor<1x10xf32>
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

