// RUN: onnx-mlir-opt --split-input-file --transfer-reduce-mean-sum-to-conv %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// Layout: NCHW [batch, channels, height, width]
// This pass converts channel-wise reduction (axis=1) to Conv operations

//===----------------------------------------------------------------------===//
// ReduceMean → Conv Tests (Channel-wise reduction)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_channel_axis_pow2
// NCHW: tensor<1x8x4x4xf32> - N=1, C=8, H=4, W=4
// Reduce axis [1] (C, power of 2) -> output tensor<1x1x4x4xf32>
func.func @reduce_mean_channel_axis_pow2(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4xf32>, tensor<1xi64>) -> tensor<1x1x4x4xf32>
    return %1 : tensor<1x1x4x4xf32>
}
// CHECK: "onnx.Conv"
// CHECK-NOT: onnx.ReduceMean

// CHECK-LABEL: @reduce_mean_channel_16
// NCHW: tensor<1x16x8x8xf32> - C=16 (power of 2)
func.func @reduce_mean_channel_16(%arg0: tensor<1x16x8x8xf32>) -> tensor<1x1x8x8xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x16x8x8xf32>, tensor<1xi64>) -> tensor<1x1x8x8xf32>
    return %1 : tensor<1x1x8x8xf32>
}
// CHECK: "onnx.Conv"
// CHECK-NOT: onnx.ReduceMean

// CHECK-LABEL: @reduce_mean_channel_negative_axis
// NCHW: tensor<1x4x4x4xf32> - reduce axis [-3] = axis 1 (C)
func.func @reduce_mean_channel_negative_axis(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %0 = onnx.Constant dense<[-3]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x4x4x4xf32>, tensor<1xi64>) -> tensor<1x1x4x4xf32>
    return %1 : tensor<1x1x4x4xf32>
}
// CHECK: "onnx.Conv"
// CHECK-NOT: onnx.ReduceMean

//===----------------------------------------------------------------------===//
// ReduceSum → Conv Tests (Channel-wise reduction)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_sum_channel_axis
// NCHW: tensor<1x8x4x4xf32> - reduce axis [1] (C)
func.func @reduce_sum_channel_axis(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4xf32>, tensor<1xi64>) -> tensor<1x1x4x4xf32>
    return %1 : tensor<1x1x4x4xf32>
}
// CHECK: "onnx.Conv"
// CHECK-NOT: onnx.ReduceSum

// CHECK-LABEL: @reduce_sum_channel_32
// NCHW: tensor<1x32x16x16xf32> - C=32
func.func @reduce_sum_channel_32(%arg0: tensor<1x32x16x16xf32>) -> tensor<1x1x16x16xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x32x16x16xf32>, tensor<1xi64>) -> tensor<1x1x16x16xf32>
    return %1 : tensor<1x1x16x16xf32>
}
// CHECK: "onnx.Conv"
// CHECK-NOT: onnx.ReduceSum

//===----------------------------------------------------------------------===//
// ReduceMean + Mul → Conv Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_mul_pattern
// ReduceMean on channel axis followed by Mul with 1/C constant
func.func @reduce_mean_mul_pattern(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4xf32>, tensor<1xi64>) -> tensor<1x1x4x4xf32>
    %2 = onnx.Constant dense<0.125> : tensor<1x1x4x4xf32>
    %3 = "onnx.Mul"(%1, %2) : (tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
    return %3 : tensor<1x1x4x4xf32>
}
// CHECK: "onnx.Conv"
// CHECK-NOT: onnx.ReduceMean
// CHECK-NOT: onnx.Mul

//===----------------------------------------------------------------------===//
// ReduceMean + Relu → Conv + Relu Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_relu_pattern
// ReduceMean on channel axis followed by Relu
func.func @reduce_mean_relu_pattern(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4xf32>, tensor<1xi64>) -> tensor<1x1x4x4xf32>
    %2 = "onnx.Relu"(%1) : (tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
    return %2 : tensor<1x1x4x4xf32>
}
// CHECK: "onnx.Conv"
// CHECK: "onnx.Relu"
// CHECK-NOT: onnx.ReduceMean

//===----------------------------------------------------------------------===//
// Negative Tests: Should NOT convert (non-channel axis)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_spatial_axis_no_convert
// NCHW: tensor<1x8x4x4xf32> - reduce axis [2] (H, not C)
// Should NOT convert because it's not channel-wise reduction
func.func @reduce_mean_spatial_axis_no_convert(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x1x4xf32> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4xf32>, tensor<1xi64>) -> tensor<1x8x1x4xf32>
    return %1 : tensor<1x8x1x4xf32>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.Conv

// CHECK-LABEL: @reduce_mean_width_axis_no_convert
// NCHW: tensor<1x8x4x4xf32> - reduce axis [3] (W, not C)
// Should NOT convert because it's not channel-wise reduction
func.func @reduce_mean_width_axis_no_convert(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x1xf32> {
    %0 = onnx.Constant dense<[3]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4xf32>, tensor<1xi64>) -> tensor<1x8x4x1xf32>
    return %1 : tensor<1x8x4x1xf32>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.Conv

// CHECK-LABEL: @reduce_mean_non_pow2_no_convert
// NCHW: tensor<1x7x4x4xf32> - C=7 (not power of 2)
// Should NOT convert because channel count is not power of 2
func.func @reduce_mean_non_pow2_no_convert(%arg0: tensor<1x7x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x7x4x4xf32>, tensor<1xi64>) -> tensor<1x1x4x4xf32>
    return %1 : tensor<1x1x4x4xf32>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.Conv

// CHECK-LABEL: @reduce_sum_spatial_axis_no_convert
// NCHW: tensor<1x8x4x4xf32> - reduce axis [2, 3] (H, W - not C)
// Should NOT convert because it's not channel-wise reduction
func.func @reduce_sum_spatial_axis_no_convert(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x1x1xf32> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4xf32>, tensor<2xi64>) -> tensor<1x8x1x1xf32>
    return %1 : tensor<1x8x1x1xf32>
}
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: onnx.Conv

//===----------------------------------------------------------------------===//
// Different batch sizes and shapes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_batch_4
// NCHW: tensor<4x16x8x8xf32> - N=4, C=16
func.func @reduce_mean_batch_4(%arg0: tensor<4x16x8x8xf32>) -> tensor<4x1x8x8xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<4x16x8x8xf32>, tensor<1xi64>) -> tensor<4x1x8x8xf32>
    return %1 : tensor<4x1x8x8xf32>
}
// CHECK: "onnx.Conv"
// CHECK-NOT: onnx.ReduceMean

// CHECK-LABEL: @reduce_sum_large_channel
// NCHW: tensor<1x64x32x32xf32> - C=64
func.func @reduce_sum_large_channel(%arg0: tensor<1x64x32x32xf32>) -> tensor<1x1x32x32xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x64x32x32xf32>, tensor<1xi64>) -> tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
}
// CHECK: "onnx.Conv"
// CHECK-NOT: onnx.ReduceSum
