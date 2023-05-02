// RUN: onnx-mlir-opt --convert-onnx-to-mhlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_reducemax_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducemax_v13
// CHECK: %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK: %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.maximum across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
}

// -----

func.func @test_reducemax_v13_keepdims(%arg0: tensor<?x20x30xf32>) -> tensor<?x1x30xf32> {
  %0 = "onnx.ReduceMaxV13"(%arg0) {axes = [1], keepdims = 1 : si64} : (tensor<?x20x30xf32>) -> tensor<?x1x30xf32>
  return %0 : tensor<?x1x30xf32>
// CHECK-LABEL:  func @test_reducemax_v13_keepdims
// CHECK-DAG: %c2 = arith.constant 2 : index
// CHECK-DAG: %c1 = arith.constant 1 : index
// CHECK-DAG: %c0 = arith.constant 0 : index
// CHECK-DAG: %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG: %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.maximum across dimensions = [1] : (tensor<?x20x30xf32>, tensor<f32>) -> tensor<?x30xf32>
// CHECK-DAG: %2 = shape.shape_of %arg0 : tensor<?x20x30xf32> -> tensor<3xindex>
// CHECK-DAG: %3 = shape.get_extent %2, %c0 : tensor<3xindex>, index -> index
// CHECK-DAG: %4 = shape.get_extent %2, %c2 : tensor<3xindex>, index -> index
// CHECK: %5 = shape.from_extents %3, %c1, %4 : index, index, index
// CHECK: %6 = shape.to_extent_tensor %5 : !shape.shape -> tensor<3xindex>
// CHECK: %7 = mhlo.dynamic_reshape %1, %6 : (tensor<?x30xf32>, tensor<3xindex>) -> tensor<?x1x30xf32>
}

// -----

func.func @test_reducemax_v13_integer_tensor(%arg0 : tensor<3x2x2xi64>) -> tensor<3x2xi64> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xi64>)-> tensor<3x2xi64>
  "func.return"(%0) : (tensor<3x2xi64>) -> ()
// CHECK-LABEL:  func @test_reducemax_v13_integer_tensor
// CHECK: %0 = mhlo.constant dense<-9223372036854775808> : tensor<i64>
// CHECK: %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.maximum across dimensions = [1] : (tensor<3x2x2xi64>, tensor<i64>) -> tensor<3x2xi64>
}

// -----

func.func @test_reducemin_v13(%arg0 : tensor<?x2x2xf32>) -> tensor<?x2xf32> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x2x2xf32>)-> tensor<?x2xf32>
  "func.return"(%0) : (tensor<?x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducemin
// CHECK: %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
// CHECK: %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.minimum across dimensions = [1] : (tensor<?x2x2xf32>, tensor<f32>) -> tensor<?x2xf32>
}

func.func @test_reducemin_v13_integer_tensor(%arg0 : tensor<?x2x2xi64>) -> tensor<?x2xi64> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x2x2xi64>)-> tensor<?x2xi64>
  "func.return"(%0) : (tensor<?x2xi64>) -> ()
// CHECK-LABEL:  func @test_reducemin_v13_integer_tensor
// CHECK: %0 = mhlo.constant dense<9223372036854775807> : tensor<i64>
// CHECK: %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.minimum across dimensions = [1] : (tensor<?x2x2xi64>, tensor<i64>) -> tensor<?x2xi64>
}

// -----

func.func @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %cst = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 ="onnx.ReduceSum"(%arg0, %cst) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducesum
// CHECK:    [[VAR_1_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:   %1 = mhlo.reduce(%arg0 init: [[VAR_1_]]) applies mhlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
}

// -----

func.func @test_reducesum_v11(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %0 ="onnx.ReduceSumV11"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducesum_v11
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
// CHECK-DAG:     [[VAR_2:%.+]] = mhlo.reshape [[VAR_1]] : (tensor<3x2xf32>) -> tensor<3x1x2xf32>
}

// -----

func.func @test_reducesum2(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
    %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
    return %0 : tensor<3x1x2xf32>
// CHECK-LABEL:  func @test_reducesum2
// CHECK-DAG:     [[VAR_0:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     [[VAR_1:%.+]] = mhlo.reduce([[PARAM_0:%.+]] init: [[VAR_0]]) applies mhlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK-DAG:     [[VAR_2:%.+]] = mhlo.reshape [[VAR_1]] : (tensor<3x2xf32>) -> tensor<3x1x2xf32>
}

func.func @test_reducemean_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<3x2xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<3x2xf32>
  "func.return"(%0) : (tensor<3x2xf32>) -> ()
// CHECK-LABEL:  func @test_reducemean_v13
// CHECK-DAG:    [[VAR_0:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:    [[VAR_1:%.+]] = mhlo.reduce([[PARAM_0:%.+]] init: [[VAR_0]]) applies mhlo.add across dimensions = [1] : (tensor<3x2x2xf32>, tensor<f32>) -> tensor<3x2xf32>
// CHECK-DAG:    [[VAR_2:%.+]] = mhlo.constant dense<2.000000e+00> : tensor<3x2xf32>
// CHECK-DAG:    [[VAR_3:%.+]] = mhlo.divide [[VAR_1]], [[VAR_2]] : tensor<3x2xf32>
// CHECK-DAG:    return [[VAR_3]] : tensor<3x2xf32>
}

func.func @test_reducemean_v13_2(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<?x?x?xf32>)-> tensor<?x?xf32>
  "func.return"(%0) : (tensor<?x?xf32>) -> ()
// CHECK-LABEL:  func @test_reducemean_v13_2
// CHECK-SAME: ([[PARAM_0:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:    [[VAR_2_:%.+]] = mhlo.reduce([[PARAM_0]] init: [[VAR_1_]]) applies mhlo.add across dimensions = [1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK-DAG:    [[VAR_3_:%.+]] = shape.shape_of [[PARAM_0]] : tensor<?x?x?xf32> -> tensor<3xindex>
// CHECK-NEXT:   [[VAR_4_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_0_]], [[VAR_3_]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x?xf32>
// CHECK-NEXT:   [[VAR_5_:%.+]] = mhlo.reduce([[VAR_4_]] init: [[VAR_1_]]) applies mhlo.add across dimensions = [1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
// CHECK-NEXT:   [[VAR_6_:%.+]] = mhlo.divide [[VAR_2_]], [[VAR_5_]] : tensor<?x?xf32>
}
