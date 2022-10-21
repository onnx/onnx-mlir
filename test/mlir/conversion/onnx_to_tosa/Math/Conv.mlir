// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x75x75xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x75x75xf32>
  return %0 : tensor<5x2x75x75xf32>
}

// -----
func.func @test_onnx_conv2d_novalue(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32>
  return %0 : tensor<5x2x965x967xf32>
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<7x3x64x64xf32>, %arg2: none) ->  tensor<5x7x74x74xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {strides = [13, 13]} : (tensor<5x3x1024x1024xf32>, tensor<7x3x64x64xf32>, none) -> tensor<5x7x74x74xf32>
  return %0 : tensor<5x7x74x74xf32>
}
// -----
func.func @test_onnx_conv2d_no_dilation_pad_stride(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x961x961xf32>{
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x961x961xf32>
  return %0 : tensor<5x2x961x961xf32>
}