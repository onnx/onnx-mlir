// RUN: onnx-mlir-opt --split-input-file --lower-reduce-to-pool %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// Layout: NCHW [batch, channels, height, width]

//===----------------------------------------------------------------------===//
// ReduceMean → AveragePool Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_spatial_hw
// NCHW: tensor<1x3x4x4> - N=1, C=3, H=4, W=4
// Reduce axes [2, 3] (H, W) -> output tensor<1x3x1x1>
func.func @reduce_mean_spatial_hw(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.AveragePool"
// CHECK-NOT: onnx.ReduceMean

// CHECK-LABEL: @reduce_mean_single_axis_h
// NCHW: tensor<1x3x8x4> - N=1, C=3, H=8, W=4
// Reduce axis [2] (H) -> output tensor<1x3x1x4>
func.func @reduce_mean_single_axis_h(%arg0: tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.AveragePool"
// CHECK-NOT: onnx.ReduceMean

// CHECK-LABEL: @reduce_mean_single_axis_w
// NCHW: tensor<1x3x4x8> - N=1, C=3, H=4, W=8
// Reduce axis [3] (W) -> output tensor<1x3x4x1>
func.func @reduce_mean_single_axis_w(%arg0: tensor<1x3x4x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x4x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[3]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x4x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x4x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x4x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.AveragePool"
// CHECK-NOT: onnx.ReduceMean

// CHECK-LABEL: @reduce_mean_negative_axis
// NCHW: tensor<1x3x4x4> - reduce axis [-2] = axis 2 (H)
func.func @reduce_mean_negative_axis(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[-2]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.AveragePool"
// CHECK-NOT: onnx.ReduceMean

//===----------------------------------------------------------------------===//
// ReduceSum → AveragePool + Mul Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_sum_spatial_hw
// NCHW: tensor<1x3x4x4> - reduce axes [2, 3] (H, W)
func.func @reduce_sum_spatial_hw(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.AveragePool"
// CHECK: "onnx.Mul"
// CHECK-NOT: onnx.ReduceSum

// CHECK-LABEL: @reduce_sum_single_axis
// NCHW: tensor<1x3x8x4> - reduce axis [2] (H)
func.func @reduce_sum_single_axis(%arg0: tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.AveragePool"
// CHECK: "onnx.Mul"
// CHECK-NOT: onnx.ReduceSum

//===----------------------------------------------------------------------===//
// ReduceMax → MaxPool (Spatial) Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_max_spatial_hw
// NCHW: tensor<1x3x4x4> - reduce axes [2, 3] (H, W)
func.func @reduce_max_spatial_hw(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.MaxPoolSingleOut"
// CHECK-NOT: onnx.ReduceMax

// CHECK-LABEL: @reduce_max_single_spatial_axis
// NCHW: tensor<1x3x8x4> - reduce axis [2] (H)
func.func @reduce_max_single_spatial_axis(%arg0: tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.MaxPoolSingleOut"
// CHECK-NOT: onnx.ReduceMax

//===----------------------------------------------------------------------===//
// ReduceMax → MaxPool (Channel via Reshape) Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_max_channel_axis
// NCHW: tensor<1x8x4x4> - reduce axis [1] (C)
func.func @reduce_max_channel_axis(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.Reshape"
// CHECK: "onnx.MaxPoolSingleOut"
// CHECK: "onnx.Reshape"
// CHECK-NOT: onnx.ReduceMax

//===----------------------------------------------------------------------===//
// Edge Cases: Trivial reduction (dimension = 1) should be simplified
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_trivial_axis
// NCHW: tensor<1x3x1x4> - reduce axis [2] where H=1 (trivial)
func.func @reduce_mean_trivial_axis(%arg0: tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// Trivial reduction (dim=1) should be replaced with input directly
// CHECK-NOT: onnx.ReduceMean
// CHECK-NOT: onnx.AveragePool
// CHECK: return %arg0

//===----------------------------------------------------------------------===//
// Different tensor shapes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_larger_spatial
// NCHW: tensor<1x64x16x16> - reduce axes [2, 3] (H, W)
func.func @reduce_mean_larger_spatial(%arg0: tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x64x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x64x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x64x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.AveragePool"
// CHECK-NOT: onnx.ReduceMean

// CHECK-LABEL: @reduce_max_larger_spatial
// NCHW: tensor<1x64x16x16> - reduce axes [2, 3] (H, W)
func.func @reduce_max_larger_spatial(%arg0: tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x64x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x64x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x64x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.MaxPoolSingleOut"
// CHECK-NOT: onnx.ReduceMax

// CHECK-LABEL: @reduce_sum_larger_spatial
// NCHW: tensor<1x32x8x8> - reduce axes [2, 3] (H, W)
func.func @reduce_sum_larger_spatial(%arg0: tensor<1x32x8x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x32x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x32x8x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x32x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x32x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.AveragePool"
// CHECK: "onnx.Mul"
// CHECK-NOT: onnx.ReduceSum

//===----------------------------------------------------------------------===//
// ReduceMeanV13 → AveragePool Tests (attribute-based axes, from
// GlobalAveragePool canonicalization)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_v13_spatial_hw
// NCHW: tensor<1x3x4x4> - N=1, C=3, H=4, W=4
// Reduce axes [2, 3] (H, W) -> output tensor<1x3x1x1>
func.func @reduce_mean_v13_spatial_hw(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %0 : tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.AveragePool"
// CHECK-NOT: onnx.ReduceMeanV13

// Channel-axis (axis=1) ReduceMean / ReduceSum is not spatial -- not converted.

// CHECK-LABEL: @reduce_mean_channel_axis_rank4_no_convert
func.func @reduce_mean_channel_axis_rank4_no_convert(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.AveragePool

// CHECK-LABEL: @reduce_sum_channel_axis_rank4_no_convert
func.func @reduce_sum_channel_axis_rank4_no_convert(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: onnx.AveragePool

// CHECK-LABEL: @reduce_mean_channel_axis_rank4_keepdims_false_no_convert
func.func @reduce_mean_channel_axis_rank4_keepdims_false_no_convert(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.AveragePool

// CHECK-LABEL: @reduce_mean_rank3_axis1_no_convert
func.func @reduce_mean_rank3_axis1_no_convert(%arg0: tensor<1x8x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.AveragePool
