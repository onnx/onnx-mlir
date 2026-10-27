// RUN: onnx-mlir-opt --split-input-file --transfer-reduce-mean-sum-to-conv %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A singleton dimension at axis 1 must not make a last-axis reduction look
// like an NCHW channel reduction. Reducing axis 1 would leave 384 values per
// batch and create an invalid reshape to the requested one-value result.
// Rank-3 spatial reductions remain native operations in this pass.

// CHECK-LABEL: @reduce_mean_singleton_channel_positive_axis
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
func.func @reduce_mean_singleton_channel_positive_axis(%arg0: tensor<72x1x384x!quant.uniform<i16:f32, 0.05:0>>) -> tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>> {
    %axes = onnx.Constant dense<[2]> : tensor<1xi64>
    %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<72x1x384x!quant.uniform<i16:f32, 0.05:0>>, tensor<1xi64>) -> tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>>
    return %0 : tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>>
}

// -----

// CHECK-LABEL: @reduce_mean_singleton_channel_negative_axis
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
func.func @reduce_mean_singleton_channel_negative_axis(%arg0: tensor<72x1x384x!quant.uniform<i16:f32, 0.05:0>>) -> tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>> {
    %axes = onnx.Constant dense<[-1]> : tensor<1xi64>
    %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<72x1x384x!quant.uniform<i16:f32, 0.05:0>>, tensor<1xi64>) -> tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>>
    return %0 : tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>>
}

// -----

// CHECK-LABEL: @reduce_mean_v13_singleton_channel_positive_axis
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: "onnx.ReduceMeanV13"
// CHECK-SAME: axes = [2]
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
func.func @reduce_mean_v13_singleton_channel_positive_axis(%arg0: tensor<72x1x384x!quant.uniform<i16:f32, 0.05:0>>) -> tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2], keepdims = 1 : si64} : (tensor<72x1x384x!quant.uniform<i16:f32, 0.05:0>>) -> tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>>
    return %0 : tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>>
}

// -----

// CHECK-LABEL: @reduce_mean_v13_singleton_channel_negative_axis
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: "onnx.ReduceMeanV13"
// CHECK-SAME: axes = [-1]
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
func.func @reduce_mean_v13_singleton_channel_negative_axis(%arg0: tensor<72x1x384x!quant.uniform<i16:f32, 0.05:0>>) -> tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1], keepdims = 1 : si64} : (tensor<72x1x384x!quant.uniform<i16:f32, 0.05:0>>) -> tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>>
    return %0 : tensor<72x1x1x!quant.uniform<i16:f32, 0.05:0>>
}

// -----

// The Mul/Relu fusion patterns use the same channel-axis check. A non-unit
// dimension at axis 1 also keeps the plain reduction pattern from matching,
// so these cases exercise the fusion patterns' own axis validation.
// CHECK-LABEL: @reduce_mean_last_axis_mul
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: "onnx.ReduceMean"
// CHECK: "onnx.Mul"
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
func.func @reduce_mean_last_axis_mul(%arg0: tensor<2x2x6xf32>) -> tensor<2x2x1xf32> {
    %axes = onnx.Constant dense<[-1]> : tensor<1xi64>
    %scale = onnx.Constant dense<0.5> : tensor<f32>
    %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x2x6xf32>, tensor<1xi64>) -> tensor<2x2x1xf32>
    %1 = "onnx.Mul"(%0, %scale) : (tensor<2x2x1xf32>, tensor<f32>) -> tensor<2x2x1xf32>
    return %1 : tensor<2x2x1xf32>
}

// -----

// CHECK-LABEL: @reduce_mean_last_axis_relu
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: "onnx.ReduceMean"
// CHECK: "onnx.Relu"
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
func.func @reduce_mean_last_axis_relu(%arg0: tensor<2x2x6xf32>) -> tensor<2x2x1xf32> {
    %axes = onnx.Constant dense<[2]> : tensor<1xi64>
    %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x2x6xf32>, tensor<1xi64>) -> tensor<2x2x1xf32>
    %1 = "onnx.Relu"(%0) : (tensor<2x2x1xf32>) -> tensor<2x2x1xf32>
    return %1 : tensor<2x2x1xf32>
}

// -----

// Supported rank-4 spatial reductions still transpose the actual reduction
// dimension to axis 1. The Conv must consume four channels, not the original
// singleton dimension at axis 1, and the inverse transpose restores the shape.
// CHECK-LABEL: @reduce_mean_v13_singleton_channel_spatial
// CHECK: "onnx.Transpose"
// CHECK-SAME: perm = [0, 3, 1, 2]
// CHECK-SAME: -> tensor<2x4x1x3x
// CHECK: "onnx.Conv"
// CHECK-SAME: tensor<1x4x1x1x
// CHECK-SAME: -> tensor<2x1x1x3x
// CHECK: "onnx.Transpose"
// CHECK-SAME: perm = [0, 2, 3, 1]
// CHECK-SAME: -> tensor<2x1x3x1x
// CHECK-NOT: "onnx.ReduceMeanV13"
// CHECK: return
func.func @reduce_mean_v13_singleton_channel_spatial(%arg0: tensor<2x1x3x4x!quant.uniform<i16:f32, 0.05:0>>) -> tensor<2x1x3x1x!quant.uniform<i16:f32, 0.05:0>> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [3], keepdims = 1 : si64} : (tensor<2x1x3x4x!quant.uniform<i16:f32, 0.05:0>>) -> tensor<2x1x3x1x!quant.uniform<i16:f32, 0.05:0>>
    return %0 : tensor<2x1x3x1x!quant.uniform<i16:f32, 0.05:0>>
}
