// RUN: onnx-mlir-opt --onnx-replace-novalue %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_novalue(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32>
  return %0 : tensor<5x2x965x967xf32>
// CHECK: %[[VAL_1:.*]] = onnx.Constant dense<0.000000e+00> : tensor<2xf32>
// CHECK: %[[VAL_2:.*]] = "onnx.Conv"(%arg0, %arg1, %[[VAL_1]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [1, 2, 3, 4]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) -> tensor<5x2x965x967xf32>
}

// -----
func.func @test_onnx_gemm_novalue(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32> {
   %none = "onnx.NoValue"() {value} : () -> none
   %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, none) -> tensor<1x4xf32>
   return %0 : tensor<1x4xf32>
// CHECK: %[[VAL_1:.*]] = onnx.Constant dense<0.000000e+00> : tensor<4xf32>
// CHECK: %[[VAL_2:.*]] = "onnx.Gemm"(%arg0, %arg1, %[[VAL_1]]) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
}

// -----
func.func @test_onnx_pad_novalue(%arg0: tensor<20x16x44x32xf32>) ->  tensor<20x16x45x33xf32>     {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 1, 1, 0, 0, 0, 0]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, none) -> tensor<20x16x45x33xf32> 
    return %2 :   tensor<20x16x45x33xf32> 
// CHECK-DAG: %[[VAL_1:.*]] = onnx.Constant dense<[0, 0, 1, 1, 0, 0, 0, 0]> : tensor<8xi64>
// CHECK-DAG: %[[VAL_2:.*]] = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
// CHECK: %[[VAL_3:.*]] = "onnx.Pad"(%arg0, %[[VAL_1]], %[[VAL_2]]) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<20x16x45x33xf32>
}

// -----
func.func @test_onnx_gemm_novalue_multiple_uses(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>,  %arg2: tensor<5x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>) {
   %none = "onnx.NoValue"() {value} : () -> none
   %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, none) -> tensor<1x4xf32>
   %1 = "onnx.Gemm"(%arg0, %arg2, %none) {transB = 0 : si64} : (tensor<1x5xf32>, tensor<5x4xf32>, none) -> tensor<1x4xf32>
   return %0, %1 : tensor<1x4xf32>, tensor<1x4xf32>
// CHECK-DAG: %[[VAL_1:.*]] = onnx.Constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-DAG: %[[VAL_2:.*]] = "onnx.Gemm"(%arg0, %arg1, %[[VAL_1]]) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
// CHECK-DAG: %[[VAL_3:.*]] = "onnx.Gemm"(%arg0, %arg2, %[[VAL_1]]) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<1x5xf32>, tensor<5x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
}