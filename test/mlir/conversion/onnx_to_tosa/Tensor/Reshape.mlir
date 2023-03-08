// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_static_reshape(%arg0: tensor<32x512x1x1xf32>) -> tensor<32x512xf32> {
  %0 = "onnx.Constant"() {value = dense<[32,512]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<32x512x1x1xf32>, tensor<2xi64>) -> tensor<32x512xf32>
  return %1 : tensor<32x512xf32>
  //CHECK:  %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = [32, 512]} : (tensor<32x512x1x1xf32>) -> tensor<32x512xf32>
}

// -----

func.func @test_static_reshape_zero(%arg0: tensor<32x512x1x1xf32>) -> tensor<32x512xf32> {
  %0 = "onnx.Constant"() {value = dense<[0,-1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<32x512x1x1xf32>, tensor<2xi64>) -> tensor<32x512xf32>
  return %1 : tensor<32x512xf32>
// CHECK-LABEL:   func.func @test_static_reshape_zero(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<32x512x1x1xf32>) -> tensor<32x512xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.reshape"(%[[VAL_0]]) {new_shape = [32, 512]} : (tensor<32x512x1x1xf32>) -> tensor<32x512xf32>
// CHECK:           return %[[VAL_2]] : tensor<32x512xf32>
}

// -----
func.func @test_dynamic_reshape(%arg0: tensor<32x512x1x1xf32>, %arg1: tensor<2xi64>) -> tensor<32x512xf32> {
  %1 = "onnx.Reshape"(%arg0, %arg1) : (tensor<32x512x1x1xf32>, tensor<2xi64>) -> tensor<32x512xf32>
  return %1 : tensor<32x512xf32>
  //CHECK-LABEL:  "onnx.Reshape"
}
// -----
func.func @test_allowzero_reshape(%arg0: tensor<32x512x1x1xf32>) -> tensor<32x512xf32> {
  %0 = "onnx.Constant"() {value = dense<[32,512]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Reshape"(%arg0, %0) {allowzero = 1 : si64}: (tensor<32x512x1x1xf32>, tensor<2xi64>) -> tensor<32x512xf32>
  return %1 : tensor<32x512xf32>
  //CHECK-LABEL:  "onnx.Reshape"
}
// -----
func.func @test_shape_zero_reshape(%arg0: tensor<32x512x1x1xf32>) -> tensor<32x512xf32> {
  %0 = "onnx.Constant"() {value = dense<[32, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<32x512x1x1xf32>, tensor<2xi64>) -> tensor<32x512xf32>
  return %1 : tensor<32x512xf32>
// CHECK-LABEL:   func.func @test_shape_zero_reshape(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<32x512x1x1xf32>) -> tensor<32x512xf32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.reshape"(%[[VAL_0]]) {new_shape = [32, 512]} : (tensor<32x512x1x1xf32>) -> tensor<32x512xf32>
// CHECK:           return %[[VAL_2]] : tensor<32x512xf32>
}
