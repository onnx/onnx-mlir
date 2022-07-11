// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s

func @test_softmax(%arg0 : tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64, onnx_opset = 13 : si64} : (tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
  "func.return"(%0) : (tensor<10x20x30xf32>) -> ()
// CHECK-LABEL:  func @test_softmax
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:   [[VAR_2_:%.+]] = mhlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies mhlo.maximum across dimensions = [1] : (tensor<10x20x30xf32>, tensor<f32>) -> tensor<10x30xf32>
// CHECK-NEXT:   [[VAR_3_:%.+]] = "mhlo.broadcast_in_dim"([[VAR_2_]]) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<10x30xf32>) -> tensor<10x20x30xf32>
// CHECK-NEXT:   [[VAR_4_:%.+]] = mhlo.subtract [[PARAM_0_]], [[VAR_3_]] : tensor<10x20x30xf32>
// CHECK-NEXT:   [[VAR_5_:%.+]] = mhlo.exponential [[VAR_4_]] : tensor<10x20x30xf32>
// CHECK-NEXT:   [[VAR_6_:%.+]] = mhlo.reduce([[VAR_5_]] init: [[VAR_0_]]) applies mhlo.add across dimensions = [1] : (tensor<10x20x30xf32>, tensor<f32>) -> tensor<10x30xf32>
// CHECK-NEXT:   [[VAR_7_:%.+]] = "mhlo.broadcast_in_dim"([[VAR_6_]]) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<10x30xf32>) -> tensor<10x20x30xf32>
// CHECK-NEXT:   [[VAR_8_:%.+]] = mhlo.divide [[VAR_5_]], [[VAR_7_]] : tensor<10x20x30xf32>
// CHECK-NEXT:   return [[VAR_8_]] : tensor<10x20x30xf32>
// CHECK-NEXT:   }
}


func @test_softmax_dynamic(%arg0 : tensor<?x20x30xf32>) -> tensor<?x20x30xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1: si64, onnx_opset = 13 : si64} : (tensor<?x20x30xf32>) -> tensor<?x20x30xf32>
  "func.return"(%0) : (tensor<?x20x30xf32>) -> ()
// CHECK-LABEL:  func @test_softmax_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20x30xf32>) -> tensor<?x20x30xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NOT:    separator of consecutive DAGs
// CHECK-DAG:    [[VAR_2_:%.+]] = mhlo.reduce([[PARAM_0_]] init: [[VAR_1_]]) applies mhlo.maximum across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-DAG:    [[VAR_3_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-NEXT:   [[VAR_4_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_2_]], [[VAR_3_]]) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<?x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-NEXT:   [[VAR_5_:%.+]] = mhlo.subtract [[PARAM_0_]], [[VAR_4_]] : tensor<?x20x30xf32>
// CHECK-NEXT:   [[VAR_6_:%.+]] = mhlo.exponential [[VAR_5_]] : tensor<?x20x30xf32>
// CHECK-NEXT:   [[VAR_7_:%.+]] = mhlo.reduce([[VAR_6_]] init: [[VAR_0_]]) applies mhlo.add across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-NEXT:   [[VAR_8_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_7_]], [[VAR_3_]]) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<?x30xf32>, tensor<3xindex>) -> tensor<?x20x30xf32>
// CHECK-NEXT:   [[VAR_9_:%.+]] = mhlo.divide [[VAR_6_]], [[VAR_8_]] : tensor<?x20x30xf32>
// CHECK-NEXT:   return [[VAR_9_]] : tensor<?x20x30xf32>
// CHECK-NEXT:   }
}