// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s -split-input-file | FileCheck %s

func.func @test_reducemax(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %0 ="onnx.ReduceMax"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducemax
// CHECK: %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK: %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.maximum across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
}

// -----

func.func @test_reducemin(%arg0 : tensor<?x2x2xf32>) -> tensor<?x2xf32> {
  %0 ="onnx.ReduceMin"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x2x2xf32>)-> tensor<?x2xf32>
  "func.return"(%0) : (tensor<?x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducemin
// CHECK: %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK: %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.minimum across dimensions = [1] : (tensor<?x2x2xf32>, tensor<f32>) -> tensor<?x2xf32>
}

// -----

func.func @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %cst = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 ="onnx.ReduceSum"(%arg0, %cst) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducesum
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %2 = mhlo.reduce(%arg0 init: [[VAR_1_]]) applies mhlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
}

// -----

func.func @test_reducesumV11(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %0 ="onnx.ReduceSumV11"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducesumV11
// CHECK: %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
}

// --
func.func @test_reducesum1(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
    %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
    return %0 : tensor<3x1x2xf32>
// CHECK-LABEL:  func @test_reducesum1
// CHECK-DAG:     [[VAR_0:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     [[VAR_1:%.+]] = mhlo.reduce([[PARAM_0:%.+]] init: [[VAR_0]]) applies mhlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK-DAG:     [[VAR_2:%.+]] = mhlo.constant dense<[3, 1, 2]> : tensor<3xi64>
// CHECK-DAG:     [[VAR_3:%.+]] = "mhlo.dynamic_reshape"([[VAR_1]], [[VAR_2]]) : (tensor<3x2xf32>, tensor<3xi64>) -> tensor<3x1x2xf32>
}

// -----

func.func @test_reducesum2(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
    %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
    return %0 : tensor<3x1x2xf32>
// CHECK-LABEL:  func @test_reducesum2
// CHECK-DAG:     [[VAR_0:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     [[VAR_1:%.+]] = mhlo.reduce([[PARAM_0:%.+]] init: [[VAR_0]]) applies mhlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK-DAG:     [[VAR_2:%.+]] = mhlo.constant dense<[3, 1, 2]> : tensor<3xi64>
// CHECK-DAG:     [[VAR_3:%.+]] = "mhlo.dynamic_reshape"([[VAR_1]], [[VAR_2]]) : (tensor<3x2xf32>, tensor<3xi64>) -> tensor<3x1x2xf32>
}

func.func @test_reducemean(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducemean
// CHECK-DAG:    [[VAR_0:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:    [[VAR_1:%.+]] = mhlo.reduce([[PARAM_0:%.+]] init: [[VAR_0]]) applies mhlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK-DAG:    [[VAR_2:%.+]] = mhlo.constant dense<2.000000e+00> : tensor<3x2xf32>
// CHECK-DAG:    [[VAR_3:%.+]] = mhlo.divide [[VAR_1]], [[VAR_2]] : tensor<3x2xf32>
// CHECK-DAG:    return [[VAR_3]] : tensor<3x2xf32>
}

func.func @test_reducemean2(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<?x?xf32>
  "func.return"(%0) : (tensor<?x?xf32>) -> ()
// CHECK-LABEL:  func @test_reducemean2
// CHECK-SAME: ([[PARAM_0:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:    [[VAR_2_:%.+]] = mhlo.reduce([[PARAM_0]] init: [[VAR_1_]]) applies mhlo.add across dimensions = [1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK-DAG:    [[VAR_3_:%.+]] = shape.shape_of [[PARAM_0]] : tensor<?x?x?xf32> -> tensor<3xindex>
// CHECK-NEXT:   [[VAR_4_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_0_]], [[VAR_3_]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x?xf32>
// CHECK-NEXT:   [[VAR_5_:%.+]] = mhlo.reduce([[VAR_4_]] init: [[VAR_1_]]) applies mhlo.add across dimensions = [1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK-NEXT:   [[VAR_6_:%.+]] = mhlo.divide [[VAR_2_]], [[VAR_5_]] : tensor<?x?xf32>
}