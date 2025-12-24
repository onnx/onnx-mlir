// RUN: onnx-mlir-opt --canonicalize %s | FileCheck %s

func.func @test_fuse_pad_avgpool(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x8x8xf32> {
    %0 = onnx.Constant dense<[0, 0, 1, 1, 0, 0, 2, 2]> : tensor<8xi64>
    %1 = onnx.Constant dense<0.000000e+00> : tensor<f32>
    %2 = "onnx.NoValue"() {value} : () -> none
    %3 = "onnx.Pad"(%arg0, %0, %1, %2) {mode = "constant"} : (tensor<1x1x4x4xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x1x7x7xf32>
    %4 = "onnx.AveragePool"(%3) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 1 : si64,
      kernel_shape = [2, 2],
      pads = [1, 1, 1, 1],
      strides = [1, 1]} : (tensor<1x1x7x7xf32>) -> tensor<1x1x8x8xf32>
    return %4 : tensor<1x1x8x8xf32>
  }


// CHECK-LABEL: func.func @test_fuse_pad_avgpool
// CHECK-NOT: onnx.Pad
// CHECK: %[[POOL:.*]] = "onnx.AveragePool"(%arg0)
// CHECK-SAME: kernel_shape = [2, 2]
// CHECK-SAME: pads = [2, 2, 3, 3]
// CHECK-SAME: strides = [1, 1]
// CHECK: return %[[POOL]]

// -----

// Test negative case: padding on batch dimension should not merge
func.func @test_no_fuse_pad_batch_dim(%arg0: tensor<1x1x4x4xf32>) -> tensor<3x1x3x3xf32> {
    %0 = onnx.Constant dense<[1, 0, 0, 0, 1, 0, 0, 0]> : tensor<8xi64>
    %1 = onnx.Constant dense<0.000000e+00> : tensor<f32>
    %2 = "onnx.NoValue"() {value} : () -> none
    %3 = "onnx.Pad"(%arg0, %0, %1, %2) {mode = "constant"} : (tensor<1x1x4x4xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<3x1x4x4xf32>
    %4 = "onnx.AveragePool"(%3) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 1 : si64,
      kernel_shape = [2, 2],
      strides = [1, 1]} : (tensor<3x1x4x4xf32>) -> tensor<3x1x3x3xf32>
    return %4 : tensor<3x1x3x3xf32>
  }

// CHECK-LABEL: func.func @test_no_fuse_pad_batch_dim
// CHECK: onnx.Pad
// CHECK: onnx.AveragePool

// -----

// Test negative case: padding on channel dimension should not merge
func.func @test_no_fuse_pad_channel_dim(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x3x3x3xf32> {
    %0 = onnx.Constant dense<[0, 1, 0, 0, 0, 1, 0, 0]> : tensor<8xi64>
    %1 = onnx.Constant dense<0.000000e+00> : tensor<f32>
    %2 = "onnx.NoValue"() {value} : () -> none
    %3 = "onnx.Pad"(%arg0, %0, %1, %2) {mode = "constant"} : (tensor<1x1x4x4xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x3x4x4xf32>
    %4 = "onnx.AveragePool"(%3) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 1 : si64,
      kernel_shape = [2, 2],
      strides = [1, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x3x3x3xf32>
    return %4 : tensor<1x3x3x3xf32>
  }

// CHECK-LABEL: func.func @test_no_fuse_pad_channel_dim
// CHECK: onnx.Pad
// CHECK: onnx.AveragePool

// -----

// Test negative case: non-zero pad value should not merge
func.func @test_no_fuse_pad_nonzero_value(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x5x5xf32> {
    %0 = onnx.Constant dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %2 = "onnx.NoValue"() {value} : () -> none
    %3 = "onnx.Pad"(%arg0, %0, %1, %2) {mode = "constant"} : (tensor<1x1x4x4xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x1x6x6xf32>
    %4 = "onnx.AveragePool"(%3) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 1 : si64,
      kernel_shape = [2, 2],
      strides = [1, 1]} : (tensor<1x1x6x6xf32>) -> tensor<1x1x5x5xf32>
    return %4 : tensor<1x1x5x5xf32>
  }

// CHECK-LABEL: func.func @test_no_fuse_pad_nonzero_value
// CHECK: onnx.Pad
// CHECK: onnx.AveragePool

// -----

// Test negative case: non-constant pad mode should not merge
func.func @test_no_fuse_pad_reflect_mode(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x5x5xf32> {
    %0 = onnx.Constant dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>
    %1 = onnx.Constant dense<0.000000e+00> : tensor<f32>
    %2 = "onnx.NoValue"() {value} : () -> none
    %3 = "onnx.Pad"(%arg0, %0, %1, %2) {mode = "reflect"} : (tensor<1x1x4x4xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x1x6x6xf32>
    %4 = "onnx.AveragePool"(%3) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 1 : si64,
      kernel_shape = [2, 2],
      strides = [1, 1]} : (tensor<1x1x6x6xf32>) -> tensor<1x1x5x5xf32>
    return %4 : tensor<1x1x5x5xf32>
  }

// CHECK-LABEL: func.func @test_no_fuse_pad_reflect_mode
// CHECK: onnx.Pad
// CHECK: onnx.AveragePool
