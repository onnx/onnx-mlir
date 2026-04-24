// RUN: onnx-mlir-opt --split-input-file --transfer-reduce-mean-sum-to-conv %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// Layout: NCHW [batch, channels, height, width]
// This pass converts channel-wise reduction (axis=1) to Conv operations

//===----------------------------------------------------------------------===//
// ReduceMean → Conv Tests (Channel-wise reduction)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_channel_axis_pow2
// NCHW: tensor<1x8x4x4> - N=1, C=8, H=4, W=4
// Reduce axis [1] (C, power of 2) -> output tensor<1x1x4x4>
func.func @reduce_mean_channel_axis_pow2(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// Defer to TransferReduceHdimToReduceCdim (quantized rank-4 axis=[1] keepdims=true).
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: "onnx.Conv"

// CHECK-LABEL: @reduce_mean_channel_16
// NCHW: tensor<1x16x8x8> - C=16 (power of 2)
func.func @reduce_mean_channel_16(%arg0: tensor<1x16x8x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x8x8x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x16x8x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x8x8x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x8x8x!quant.uniform<i8:f32, 0.05:0>>
}
// Defer to TransferReduceHdimToReduceCdim (quantized rank-4 axis=[1] keepdims=true).
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: "onnx.Conv"

// CHECK-LABEL: @reduce_mean_channel_negative_axis
// NCHW: tensor<1x4x4x4> - reduce axis [-3] = axis 1 (C)
func.func @reduce_mean_channel_negative_axis(%arg0: tensor<1x4x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[-3]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x4x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// Defer to TransferReduceHdimToReduceCdim (quantized rank-4 axis=[1] keepdims=true).
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: "onnx.Conv"

//===----------------------------------------------------------------------===//
// ReduceSum → Conv Tests (Channel-wise reduction)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_sum_channel_axis
// NCHW: tensor<1x8x4x4> - reduce axis [1] (C)
func.func @reduce_sum_channel_axis(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// Defer to TransferReduceHdimToReduceCdim (quantized rank-4 axis=[1] keepdims=true).
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: "onnx.Conv"

// CHECK-LABEL: @reduce_sum_channel_32
// NCHW: tensor<1x32x16x16> - C=32
func.func @reduce_sum_channel_32(%arg0: tensor<1x32x16x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x16x16x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x32x16x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x16x16x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x16x16x!quant.uniform<i8:f32, 0.05:0>>
}
// Defer to TransferReduceHdimToReduceCdim (quantized rank-4 axis=[1] keepdims=true).
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: "onnx.Conv"

//===----------------------------------------------------------------------===//
// ReduceMean Spatial Axis → Conv Tests (non-channel axis via transpose)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_spatial_axis2
// NCHW: tensor<1x8x4x4> - reduce axis [2] (H=4, power of 2)
// Converted by ReduceMeanSpatialAxisToConvPattern via transpose
func.func @reduce_mean_spatial_axis2(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x8x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x8x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.Transpose"
// CHECK: "onnx.Conv"
// CHECK: "onnx.Transpose"
// CHECK-NOT: onnx.ReduceMean

// CHECK-LABEL: @reduce_mean_spatial_axis3
// NCHW: tensor<1x8x4x4> - reduce axis [3] (W=4, power of 2)
// Converted by ReduceMeanSpatialAxisToConvPattern via transpose
func.func @reduce_mean_spatial_axis3(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x4x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[3]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x8x4x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x8x4x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.Transpose"
// CHECK: "onnx.Conv"
// CHECK: "onnx.Transpose"
// CHECK-NOT: onnx.ReduceMean

// CHECK-LABEL: @reduce_mean_spatial_non_pow2_no_convert
// NCHW: tensor<1x8x5x4> - reduce axis [2] (H=5, not power of 2)
// Should NOT convert because ReduceMeanSpatial requires power-of-2
func.func @reduce_mean_spatial_non_pow2_no_convert(%arg0: tensor<1x8x5x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x5x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x8x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x8x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.Conv

// CHECK-LABEL: @reduce_mean_non_pow2_no_convert
// NCHW: tensor<1x7x4x4> - C=7 (not power of 2)
// Should NOT convert because channel count is not power of 2
func.func @reduce_mean_non_pow2_no_convert(%arg0: tensor<1x7x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x7x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.Conv

//===----------------------------------------------------------------------===//
// ReduceSum Spatial Axis → Conv Tests (non-channel axis via transpose)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_sum_spatial_axis2_keepdims
// NCHW: tensor<1x8x4x4> - reduce axis [2] (H)
// Transposes H to channel position, Conv reduces, transpose back
func.func @reduce_sum_spatial_axis2_keepdims(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x8x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x8x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.Transpose"
// CHECK: "onnx.Conv"
// CHECK: "onnx.Transpose"
// CHECK-NOT: onnx.ReduceSum

// CHECK-LABEL: @reduce_sum_spatial_batch_axis_no_convert
// NCHW: tensor<1x8x4x4> - reduce axis [0] (batch)
// Should NOT convert because batch axis is rejected
func.func @reduce_sum_spatial_batch_axis_no_convert(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[0]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: onnx.Conv

// CHECK-LABEL: @reduce_sum_spatial_multi_axis_no_convert
// NCHW: tensor<1x8x4x4> - reduce axis [2, 3] (H, W - multi-axis rejected)
// Should NOT convert because axes.size() != 1
func.func @reduce_sum_spatial_multi_axis_no_convert(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x8x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x8x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: onnx.Conv

//===----------------------------------------------------------------------===//
// Different batch sizes and shapes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_batch_4
// NCHW: tensor<4x16x8x8> - N=4, C=16
func.func @reduce_mean_batch_4(%arg0: tensor<4x16x8x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<4x1x8x8x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<4x16x8x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<4x1x8x8x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<4x1x8x8x!quant.uniform<i8:f32, 0.05:0>>
}
// Defer to TransferReduceHdimToReduceCdim (quantized rank-4 axis=[1] keepdims=true).
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: "onnx.Conv"

// CHECK-LABEL: @reduce_sum_large_channel
// NCHW: tensor<1x64x32x32> - C=64
func.func @reduce_sum_large_channel(%arg0: tensor<1x64x32x32x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x32x32x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x64x32x32x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x32x32x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x32x32x!quant.uniform<i8:f32, 0.05:0>>
}
// Defer to TransferReduceHdimToReduceCdim (quantized rank-4 axis=[1] keepdims=true).
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: "onnx.Conv"

// -----

//===----------------------------------------------------------------------===//
// Negative Tests: Integer element types (should NOT convert)
// Conv only supports float/quantized types, not integer types like i64.
// These patterns arise from Cast/IsNaN counting in instance normalization.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_i64_input_no_convert
// NCHW: tensor<1x8x4x4xi64> - integer element type
// Should NOT convert because onnx.Conv does not support i64
func.func @reduce_mean_i64_input_no_convert(%arg0: tensor<1x8x4x4xi64>) -> tensor<1x1x4x4xi64> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4xi64>, tensor<1xi64>) -> tensor<1x1x4x4xi64>
    return %1 : tensor<1x1x4x4xi64>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.Conv

// -----

// CHECK-LABEL: @reduce_sum_i64_input_no_convert
// NCHW: tensor<1x8x4x4xi64> - integer element type
// Should NOT convert because onnx.Conv does not support i64
func.func @reduce_sum_i64_input_no_convert(%arg0: tensor<1x8x4x4xi64>) -> tensor<1x1x4x4xi64> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4xi64>, tensor<1xi64>) -> tensor<1x1x4x4xi64>
    return %1 : tensor<1x1x4x4xi64>
}
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: onnx.Conv

// -----

// CHECK-LABEL: @reduce_mean_i32_input_no_convert
// NCHW: tensor<1x4x4x4xi32> - integer element type
// Should NOT convert because onnx.Conv does not support i32
func.func @reduce_mean_i32_input_no_convert(%arg0: tensor<1x4x4x4xi32>) -> tensor<1x1x4x4xi32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x4x4x4xi32>, tensor<1xi64>) -> tensor<1x1x4x4xi32>
    return %1 : tensor<1x1x4x4xi32>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.Conv

// -----

// CHECK-LABEL: @reduce_mean_f32_input_should_convert
// NCHW: tensor<1x8x4x4xf32> - float element type
// Should convert because onnx.Conv supports f32
func.func @reduce_mean_f32_input_should_convert(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4xf32>, tensor<1xi64>) -> tensor<1x1x4x4xf32>
    return %1 : tensor<1x1x4x4xf32>
}
// CHECK: "onnx.Conv"
// CHECK-NOT: onnx.ReduceMean
