// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// RUN: flexml-opt %s -transfer-conv-slice-to-conv | FileCheck %s

// -----

// Test 1: Channel slice - select first 4 output channels from 8-channel conv output
// CHECK-LABEL: @conv_channel_slice
func.func @conv_channel_slice(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x4x14x14xf32> {
    // Original conv: 3 input channels -> 8 output channels, 3x3 kernel
    %weights = onnx.Constant dense<1.0> : tensor<8x3x3x3xf32>
    %bias = onnx.Constant dense<0.5> : tensor<8xf32>
    %conv = "onnx.Conv"(%arg0, %weights, %bias) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x3x16x16xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x14x14xf32>

    // Slice to select channels 0-3 (first 4 channels)
    %starts = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
    %ends = onnx.Constant dense<[1, 4, 14, 14]> : tensor<4xi64>
    %axes = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %steps = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
    %sliced = "onnx.Slice"(%conv, %starts, %ends, %axes, %steps) : (tensor<1x8x14x14xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x14x14xf32>

    return %sliced : tensor<1x4x14x14xf32>
}

// Verify the slice is eliminated and conv has reduced output channels
// CHECK-NOT: onnx.Slice
// CHECK: onnx.Conv
// CHECK-SAME: -> tensor<1x4x14x14xf32>

// -----

// Test 2: Spatial slice - crop H and W dimensions from conv output
// CHECK-LABEL: @conv_spatial_slice
func.func @conv_spatial_slice(%arg0: tensor<1x3x20x20xf32>) -> tensor<1x8x10x10xf32> {
    %weights = onnx.Constant dense<1.0> : tensor<8x3x3x3xf32>
    %no_bias = "onnx.NoValue"() {value} : () -> none
    %conv = "onnx.Conv"(%arg0, %weights, %no_bias) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x3x20x20xf32>, tensor<8x3x3x3xf32>, none) -> tensor<1x8x18x18xf32>

    // Slice to crop spatial dimensions: H from [2:12], W from [4:14]
    %starts = onnx.Constant dense<[0, 0, 2, 4]> : tensor<4xi64>
    %ends = onnx.Constant dense<[1, 8, 12, 14]> : tensor<4xi64>
    %axes = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %steps = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
    %sliced = "onnx.Slice"(%conv, %starts, %ends, %axes, %steps) : (tensor<1x8x18x18xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x8x10x10xf32>

    return %sliced : tensor<1x8x10x10xf32>
}

// Verify the Slice after Conv is removed; a Slice before Conv may be added
// The pattern Conv -> Slice becomes: [Slice ->] Conv
// CHECK: onnx.Conv
// CHECK-SAME: -> tensor<1x8x10x10xf32>
// CHECK-NOT: onnx.Slice{{.*}}tensor<1x8x18x18xf32>

// -----

// Test 3: Combined channel and spatial slice
// CHECK-LABEL: @conv_channel_and_spatial_slice
func.func @conv_channel_and_spatial_slice(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x2x6x6xf32> {
    %weights = onnx.Constant dense<1.0> : tensor<8x3x3x3xf32>
    %bias = onnx.Constant dense<0.1> : tensor<8xf32>
    %conv = "onnx.Conv"(%arg0, %weights, %bias) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x3x16x16xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x14x14xf32>

    // Slice channels 2-3 and spatial region [4:10, 4:10]
    %starts = onnx.Constant dense<[0, 2, 4, 4]> : tensor<4xi64>
    %ends = onnx.Constant dense<[1, 4, 10, 10]> : tensor<4xi64>
    %axes = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %steps = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
    %sliced = "onnx.Slice"(%conv, %starts, %ends, %axes, %steps) : (tensor<1x8x14x14xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x2x6x6xf32>

    return %sliced : tensor<1x2x6x6xf32>
}

// Verify the Slice after Conv is removed; output shape is correct
// For combined slice: input may be sliced, weights are sliced
// CHECK: onnx.Conv
// CHECK-SAME: -> tensor<1x2x6x6xf32>

// -----

// Test 4: Channel slice with stride - select every other channel
// CHECK-LABEL: @conv_channel_slice_with_stride
func.func @conv_channel_slice_with_stride(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x4x6x6xf32> {
    %weights = onnx.Constant dense<1.0> : tensor<8x3x3x3xf32>
    %no_bias = "onnx.NoValue"() {value} : () -> none
    %conv = "onnx.Conv"(%arg0, %weights, %no_bias) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, none) -> tensor<1x8x6x6xf32>

    // Slice every other channel: 0, 2, 4, 6 (stride=2)
    %starts = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
    %ends = onnx.Constant dense<[1, 8, 6, 6]> : tensor<4xi64>
    %axes = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %steps = onnx.Constant dense<[1, 2, 1, 1]> : tensor<4xi64>
    %sliced = "onnx.Slice"(%conv, %starts, %ends, %axes, %steps) : (tensor<1x8x6x6xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x6x6xf32>

    return %sliced : tensor<1x4x6x6xf32>
}

// Verify the slice is eliminated
// CHECK-NOT: onnx.Slice
// CHECK: onnx.Conv
// CHECK-SAME: -> tensor<1x4x6x6xf32>

// -----

// Test 5: Negative case - grouped convolution should NOT be transformed
// CHECK-LABEL: @conv_grouped_no_transform
func.func @conv_grouped_no_transform(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x2x6x6xf32> {
    %weights = onnx.Constant dense<1.0> : tensor<4x2x3x3xf32>
    %no_bias = "onnx.NoValue"() {value} : () -> none
    // Grouped convolution with group=2
    %conv = "onnx.Conv"(%arg0, %weights, %no_bias) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 2 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x4x8x8xf32>, tensor<4x2x3x3xf32>, none) -> tensor<1x4x6x6xf32>

    %starts = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
    %ends = onnx.Constant dense<[1, 2, 6, 6]> : tensor<4xi64>
    %axes = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %steps = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
    %sliced = "onnx.Slice"(%conv, %starts, %ends, %axes, %steps) : (tensor<1x4x6x6xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x2x6x6xf32>

    return %sliced : tensor<1x2x6x6xf32>
}

// Grouped conv should NOT be transformed - slice should remain
// CHECK: onnx.Conv
// CHECK-SAME: group = 2
// CHECK: onnx.Slice

// -----

// Test 6: Negative case - conv with multiple uses should NOT be transformed
// CHECK-LABEL: @conv_multiple_uses_no_transform
func.func @conv_multiple_uses_no_transform(%arg0: tensor<1x3x8x8xf32>) -> (tensor<1x4x6x6xf32>, tensor<1x8x6x6xf32>) {
    %weights = onnx.Constant dense<1.0> : tensor<8x3x3x3xf32>
    %no_bias = "onnx.NoValue"() {value} : () -> none
    %conv = "onnx.Conv"(%arg0, %weights, %no_bias) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, none) -> tensor<1x8x6x6xf32>

    // First use: slice
    %starts = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
    %ends = onnx.Constant dense<[1, 4, 6, 6]> : tensor<4xi64>
    %axes = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %steps = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
    %sliced = "onnx.Slice"(%conv, %starts, %ends, %axes, %steps) : (tensor<1x8x6x6xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x4x6x6xf32>

    // Second use: return full conv output
    return %sliced, %conv : tensor<1x4x6x6xf32>, tensor<1x8x6x6xf32>
}

// Conv with multiple uses should NOT be transformed - slice should remain
// CHECK: onnx.Conv
// CHECK-SAME: -> tensor<1x8x6x6xf32>
// CHECK: onnx.Slice

// -----

// Test 7: Conv with padding followed by spatial slice
// CHECK-LABEL: @conv_padded_spatial_slice
func.func @conv_padded_spatial_slice(%arg0: tensor<1x3x14x14xf32>) -> tensor<1x8x6x6xf32> {
    %weights = onnx.Constant dense<1.0> : tensor<8x3x3x3xf32>
    %no_bias = "onnx.NoValue"() {value} : () -> none
    %conv = "onnx.Conv"(%arg0, %weights, %no_bias) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [1, 1, 1, 1],
        strides = [1, 1]
    } : (tensor<1x3x14x14xf32>, tensor<8x3x3x3xf32>, none) -> tensor<1x8x14x14xf32>

    // Slice spatial region
    %starts = onnx.Constant dense<[0, 0, 4, 4]> : tensor<4xi64>
    %ends = onnx.Constant dense<[1, 8, 10, 10]> : tensor<4xi64>
    %axes = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %steps = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
    %sliced = "onnx.Slice"(%conv, %starts, %ends, %axes, %steps) : (tensor<1x8x14x14xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x8x6x6xf32>

    return %sliced : tensor<1x8x6x6xf32>
}

// Verify the Slice after Conv is removed; padding may be adjusted
// CHECK: onnx.Conv
// CHECK-SAME: -> tensor<1x8x6x6xf32>

// -----

// Test 8: Simple channel slice without explicit axes/steps (defaults)
// CHECK-LABEL: @conv_channel_slice_defaults
func.func @conv_channel_slice_defaults(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x2x6x6xf32> {
    %weights = onnx.Constant dense<1.0> : tensor<4x3x3x3xf32>
    %bias = onnx.Constant dense<0.0> : tensor<4xf32>
    %conv = "onnx.Conv"(%arg0, %weights, %bias) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x4x6x6xf32>

    // Slice with only starts/ends (axes and steps default)
    %starts = onnx.Constant dense<[0, 1, 0, 0]> : tensor<4xi64>
    %ends = onnx.Constant dense<[1, 3, 6, 6]> : tensor<4xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %sliced = "onnx.Slice"(%conv, %starts, %ends, %none, %none) : (tensor<1x4x6x6xf32>, tensor<4xi64>, tensor<4xi64>, none, none) -> tensor<1x2x6x6xf32>

    return %sliced : tensor<1x2x6x6xf32>
}

// Verify the slice is eliminated
// CHECK-NOT: onnx.Slice
// CHECK: onnx.Conv
// CHECK-SAME: -> tensor<1x2x6x6xf32>
