// RUN: onnx-mlir-opt --shape-inference --canonicalize %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test the folder called in canonicalize pass.
/// Shape-inference is invoked to ensure the type is correct and
/// constant may be introduced.
//===----------------------------------------------------------------------===//

// -----

func.func @test_squeeze() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[[4.0]], [[16.0]]]> : tensor<2x1x1xf32>
  %1 = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %2 = "onnx.Squeeze"(%0, %1) : (tensor<2x1x1xf32>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_squeeze
// CHECK-SAME:   () -> tensor<2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[4.000000e+00, 1.600000e+01]> : tensor<2xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2xf32>
}

// -----

func.func @test_squeezev11() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[[4.0]], [[16.0]]]> : tensor<2x1x1xf32>
  %1 = "onnx.SqueezeV11"(%0) {axes = [1, 2]} : (tensor<2x1x1xf32>) -> tensor<*xf32>
  "onnx.Return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_squeezev11
// CHECK-SAME:   () -> tensor<2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[4.000000e+00, 1.600000e+01]> : tensor<2xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2xf32>
}


// -----

func.func @test_reduceMeanIsNoopWithEmptyAxes(%arg0: tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.ReduceMean"(%arg0, %0) {noop_with_empty_axes = 1: si64} : (tensor<4x512x256x8xf32>, none) -> tensor<4x512x256x8xf32>
  return %1 : tensor<4x512x256x8xf32>
}

// CHECK-LABEL: @test_reduceMeanIsNoopWithEmptyAxes
// CHECK-SAME: (%[[VAL_0:.*]]: tensor<4x512x256x8xf32>) -> tensor<4x512x256x8xf32> {
// CHECK:   return %[[VAL_0]] : tensor<4x512x256x8xf32>
// CHECK: }

// -----

func.func @test_slice(%arg0: tensor<16x1x2500x4xf32>) -> tensor<16x1x2500x4xf32> {
  %0 = onnx.Constant dense<0> : tensor<1xi64>
  %1 = onnx.Constant dense<4> : tensor<1xi64>
  %2 = onnx.Constant dense<3> : tensor<1xi64>
  %3 = onnx.Constant dense<1> : tensor<1xi64>
  %4 = "onnx.Slice"(%arg0, %0, %1, %2, %3) : (tensor<16x1x2500x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<16x1x2500x4xf32>
  return %4 : tensor<16x1x2500x4xf32>
}

// CHECK-LABEL: @test_slice
// CHECK-SAME: (%[[VAL_0:.*]]: tensor<16x1x2500x4xf32>) -> tensor<16x1x2500x4xf32> {
// CHECK:   return %[[VAL_0]] : tensor<16x1x2500x4xf32>
// CHECK: }

// -----

func.func @test_slice_reverse_tensor(%arg0: tensor<3x2xi64>) -> tensor<3x2xi64> {
  %0 = onnx.Constant dense<-1> : tensor<1xi64>
  %1 = onnx.Constant dense<-9223372036854775807> : tensor<1xi64>
  %2 = onnx.Constant dense<0> : tensor<1xi64>
  %3 = onnx.Constant dense<-1> : tensor<1xi64>
  %4 = "onnx.Slice"(%arg0, %0, %1, %2, %3) {onnx_node_name = "/Slice"} : (tensor<3x2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x2xi64>
  return %4 : tensor<3x2xi64>
}

// CHECK-LABEL:  func.func @test_slice_reverse_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xi64>) -> tensor<3x2xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-9223372036854775807> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {onnx_node_name = "/Slice"} : (tensor<3x2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x2xi64>
// CHECK:           return [[VAR_3_]] : tensor<3x2xi64>