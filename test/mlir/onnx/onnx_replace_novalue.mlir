// RUN: onnx-mlir-opt --shape-inference --onnx-replace-novalue %s -split-input-file | FileCheck %s

// Simple intro of layout transform

func.func @test_onnx_conv2d_novalue(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32>
  return %0 : tensor<5x2x965x967xf32>
// CHECK:           %[[VAL_1:.*]] = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK:           %[[VAL_2:.*]] = "onnx.Conv"(%arg0, %arg1, %[[VAL_1]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [1, 2, 3, 4]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) -> tensor<5x2x965x967xf32>
 }
