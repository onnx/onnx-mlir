// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=onnx.Conv' %s -split-input-file | FileCheck %s

// -----

// Test Conv: Basic case with Stride=1, Padding=0, Dilation=1
func.func @conv_basic(%arg0: tensor<1x3x5x5xf32>, %arg1: tensor<2x3x3x3xf32>)
    -> tensor<1x2x3x3xf32> {
  %none = "onnx.NoValue"() : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [0, 0, 0, 0],
    strides = [1, 1]
  } : (tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>, none) -> tensor<1x2x3x3xf32>
  return %0 : tensor<1x2x3x3xf32>

  // CHECK-LABEL: conv_basic
  // CHECK-DAG: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[EMPTY:%.+]] = tensor.empty() : tensor<1x2x3x3xf32>
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY]] : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
  // CHECK: [[RESULT:%.+]] = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>) outs([[FILLED]] : tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
  // CHECK: return [[RESULT]] : tensor<1x2x3x3xf32>
}

// -----

// Test Conv: Stride=2 case
func.func @conv_stride2(%arg0: tensor<1x3x10x10xf32>, %arg1: tensor<2x3x3x3xf32>)
    -> tensor<1x2x4x4xf32> {
  %none = "onnx.NoValue"() : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [0, 0, 0, 0],
    strides = [2, 2]
  } : (tensor<1x3x10x10xf32>, tensor<2x3x3x3xf32>, none) -> tensor<1x2x4x4xf32>
  return %0 : tensor<1x2x4x4xf32>

  // CHECK-LABEL: conv_stride2
  // CHECK-DAG: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[EMPTY:%.+]] = tensor.empty() : tensor<1x2x4x4xf32>
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPTY]] : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  // CHECK: [[RESULT:%.+]] = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x3x10x10xf32>, tensor<2x3x3x3xf32>) outs([[FILLED]] : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  // CHECK: return [[RESULT]] : tensor<1x2x4x4xf32>
}

// -----

// Test Conv: Reject case with Padding=1 (should NOT be lowered)
func.func @conv_reject_padding(%arg0: tensor<1x3x5x5xf32>,
    %arg1: tensor<2x3x3x3xf32>) -> tensor<1x2x5x5xf32> {
  %none = "onnx.NoValue"() : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [1, 1, 1, 1],
    strides = [1, 1]
  } : (tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>, none) -> tensor<1x2x5x5xf32>
  return %0 : tensor<1x2x5x5xf32>

  // CHECK-LABEL: conv_reject_padding
  // CHECK-NOT: linalg.conv_2d_nchw_fchw
  // CHECK: %{{.+}} = "onnx.Conv"(%arg0, %arg1, %{{.+}}) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [1, 1, 1, 1], strides = [1, 1]}> : (tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>, none) -> tensor<1x2x5x5xf32>
}

// -----

// Test Conv: Reject case with Bias (should NOT be lowered)
func.func @conv_reject_bias(%arg0: tensor<1x3x5x5xf32>, %arg1: tensor<2x3x3x3xf32>,
    %arg2: tensor<2xf32>) -> tensor<1x2x3x3xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [0, 0, 0, 0],
    strides = [1, 1]
  } : (tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>, tensor<2xf32>)
      -> tensor<1x2x3x3xf32>
  return %0 : tensor<1x2x3x3xf32>

  // CHECK-LABEL: conv_reject_bias
  // CHECK-NOT: linalg.conv_2d_nchw_fchw
  // CHECK: %{{.+}} = "onnx.Conv"(%arg0, %arg1, %arg2) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>, tensor<2xf32>) -> tensor<1x2x3x3xf32>
}

// -----

// Test Conv: Reject case with Dilation > 1 (should NOT be lowered)
func.func @conv_reject_dilation(%arg0: tensor<1x3x5x5xf32>,
    %arg1: tensor<2x3x3x3xf32>) -> tensor<1x2x3x3xf32> {
  %none = "onnx.NoValue"() : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    dilations = [2, 2],
    group = 1 : si64,
    pads = [0, 0, 0, 0],
    strides = [1, 1]
  } : (tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>, none) -> tensor<1x2x3x3xf32>
  return %0 : tensor<1x2x3x3xf32>

  // CHECK-LABEL: conv_reject_dilation
  // CHECK-NOT: linalg.conv_2d_nchw_fchw
  // CHECK: %{{.+}} = "onnx.Conv"(%arg0, %arg1, %{{.+}}) <{auto_pad = "NOTSET", dilations = [2, 2], group = 1 : si64, pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x5x5xf32>, tensor<2x3x3x3xf32>, none) -> tensor<1x2x3x3xf32>
}

