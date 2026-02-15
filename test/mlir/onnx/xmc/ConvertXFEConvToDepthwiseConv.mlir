// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// RUN: onnx-mlir-opt --convert-xfe-conv-to-depthwise-conv %s --split-input-file | FileCheck %s

// =============================================================================
// Test 1: Basic 2D depthwise conv (group == input_channels) - SHOULD CONVERT
// =============================================================================
// Input X: NHWC [N=1, H=56, W=56, C=64]
// Weight W: OHWI [C_out=64, kH=3, kW=3, C_in/group=1]
// group = 64 (matches input channels) -> depthwise conv

// CHECK-LABEL: @depthwise_conv2d_basic
func.func @depthwise_conv2d_basic(%arg0: tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x54x54x64x!quant.uniform<i8:f32, 0.1:0>> {
    // Weight in OHWI format: [C_out, kH, kW, C_in/group] = [64, 3, 3, 1]
    %weights = onnx.Constant {value = dense<1> : tensor<64x3x3x1xi8>} : tensor<64x3x3x1x!quant.uniform<i8:f32, 0.05:0>>
    %none = "onnx.NoValue"() {value} : () -> none
    
    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 64 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>, tensor<64x3x3x1x!quant.uniform<i8:f32, 0.05:0>>, none) -> tensor<1x54x54x64x!quant.uniform<i8:f32, 0.1:0>>
    
    return %conv : tensor<1x54x54x64x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: kernel_shape = [3, 3]

// -----

// =============================================================================
// Test 2: 2D depthwise conv with bias - SHOULD CONVERT
// =============================================================================
// CHECK-LABEL: @depthwise_conv2d_with_bias
func.func @depthwise_conv2d_with_bias(%arg0: tensor<1x28x28x32x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x28x28x32x!quant.uniform<i8:f32, 0.1:0>> {
    // Weight in OHWI format: [C_out=32, kH=3, kW=3, C_in/group=1]
    %weights = onnx.Constant {value = dense<1> : tensor<32x3x3x1xi8>} : tensor<32x3x3x1x!quant.uniform<i8:f32, 0.05:0>>
    %bias = onnx.Constant {value = dense<0> : tensor<32xi32>} : tensor<32x!quant.uniform<i32:f32, 0.005:0>>
    
    %conv = "onnx.XFEConv"(%arg0, %weights, %bias) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 32 : si64,
        kernel_shape = [3, 3],
        pads = [1, 1, 1, 1],
        strides = [1, 1]
    } : (tensor<1x28x28x32x!quant.uniform<i8:f32, 0.1:0>>, tensor<32x3x3x1x!quant.uniform<i8:f32, 0.05:0>>, tensor<32x!quant.uniform<i32:f32, 0.005:0>>) -> tensor<1x28x28x32x!quant.uniform<i8:f32, 0.1:0>>
    
    return %conv : tensor<1x28x28x32x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: auto_pad = "SAME_UPPER"

// -----

// =============================================================================
// Test 3: 2D depthwise conv with strides - SHOULD CONVERT
// =============================================================================
// CHECK-LABEL: @depthwise_conv2d_strided
func.func @depthwise_conv2d_strided(%arg0: tensor<1x112x112x64x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x56x56x64x!quant.uniform<u8:f32, 0.1:128>> {
    // Weight in OHWI format: [C_out=64, kH=3, kW=3, C_in/group=1]
    %weights = onnx.Constant {value = dense<1> : tensor<64x3x3x1xi8>} : tensor<64x3x3x1x!quant.uniform<i8:f32, 0.05:0>>
    %none = "onnx.NoValue"() {value} : () -> none
    
    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 64 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 1, 1],
        strides = [2, 2]
    } : (tensor<1x112x112x64x!quant.uniform<u8:f32, 0.1:128>>, tensor<64x3x3x1x!quant.uniform<i8:f32, 0.05:0>>, none) -> tensor<1x56x56x64x!quant.uniform<u8:f32, 0.1:128>>
    
    return %conv : tensor<1x56x56x64x!quant.uniform<u8:f32, 0.1:128>>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: strides = [2, 2]

// -----

// =============================================================================
// Test 4: 2D depthwise conv with dilations - SHOULD CONVERT
// =============================================================================
// CHECK-LABEL: @depthwise_conv2d_dilated
func.func @depthwise_conv2d_dilated(%arg0: tensor<1x64x64x128x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x60x60x128x!quant.uniform<i8:f32, 0.1:0>> {
    // Weight in OHWI format: [C_out=128, kH=3, kW=3, C_in/group=1]
    %weights = onnx.Constant {value = dense<1> : tensor<128x3x3x1xi8>} : tensor<128x3x3x1x!quant.uniform<i8:f32, 0.05:0>>
    %none = "onnx.NoValue"() {value} : () -> none
    
    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "NOTSET",
        dilations = [2, 2],
        group = 128 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x64x64x128x!quant.uniform<i8:f32, 0.1:0>>, tensor<128x3x3x1x!quant.uniform<i8:f32, 0.05:0>>, none) -> tensor<1x60x60x128x!quant.uniform<i8:f32, 0.1:0>>
    
    return %conv : tensor<1x60x60x128x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: dilations = [2, 2]

// -----

// =============================================================================
// Test 5: Regular conv (group = 1) - SHOULD NOT CONVERT
// =============================================================================
// This is a standard convolution, not depthwise
// CHECK-LABEL: @regular_conv_not_depthwise
func.func @regular_conv_not_depthwise(%arg0: tensor<1x56x56x3x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>> {
    // Weight in OHWI format: [C_out=64, kH=3, kW=3, C_in/group=3]
    %weights = onnx.Constant {value = dense<1> : tensor<64x3x3x3xi8>} : tensor<64x3x3x3x!quant.uniform<i8:f32, 0.05:0>>
    %none = "onnx.NoValue"() {value} : () -> none
    
    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [1, 1, 1, 1],
        strides = [1, 1]
    } : (tensor<1x56x56x3x!quant.uniform<i8:f32, 0.1:0>>, tensor<64x3x3x3x!quant.uniform<i8:f32, 0.05:0>>, none) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>
    
    return %conv : tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK-NOT: onnx.XCOMPILERDepthwiseConv
// CHECK: onnx.XFEConv

// -----

// =============================================================================
// Test 6: Grouped conv (group > 1 but != input_channels) - SHOULD NOT CONVERT
// =============================================================================
// This is a grouped convolution with 2 groups, but 64 channels (not depthwise)
// CHECK-LABEL: @grouped_conv_not_depthwise
func.func @grouped_conv_not_depthwise(%arg0: tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>> {
    // Weight in OHWI format: [C_out=64, kH=3, kW=3, C_in/group=32] (group=2, so 64/2=32)
    %weights = onnx.Constant {value = dense<1> : tensor<64x3x3x32xi8>} : tensor<64x3x3x32x!quant.uniform<i8:f32, 0.05:0>>
    %none = "onnx.NoValue"() {value} : () -> none
    
    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 2 : si64,
        kernel_shape = [3, 3],
        pads = [1, 1, 1, 1],
        strides = [1, 1]
    } : (tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>, tensor<64x3x3x32x!quant.uniform<i8:f32, 0.05:0>>, none) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>
    
    return %conv : tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK-NOT: onnx.XCOMPILERDepthwiseConv
// CHECK: onnx.XFEConv

// -----

// =============================================================================
// Test 7: 3D depthwise conv (5D tensors) - SHOULD CONVERT
// =============================================================================
// Input X: NDHWC [N=1, D=16, H=32, W=32, C=32]
// Weight W: ODHWI [C_out=32, kD=3, kH=3, kW=3, C_in/group=1]
// group = 32 (matches input channels)

// CHECK-LABEL: @depthwise_conv3d_basic
func.func @depthwise_conv3d_basic(%arg0: tensor<1x16x32x32x32x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x14x30x30x32x!quant.uniform<i8:f32, 0.1:0>> {
    // Weight in ODHWI format: [C_out=32, kD=3, kH=3, kW=3, C_in/group=1]
    %weights = onnx.Constant {value = dense<1> : tensor<32x3x3x3x1xi8>} : tensor<32x3x3x3x1x!quant.uniform<i8:f32, 0.05:0>>
    %none = "onnx.NoValue"() {value} : () -> none
    
    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "NOTSET",
        dilations = [1, 1, 1],
        group = 32 : si64,
        kernel_shape = [3, 3, 3],
        pads = [0, 0, 0, 0, 0, 0],
        strides = [1, 1, 1]
    } : (tensor<1x16x32x32x32x!quant.uniform<i8:f32, 0.1:0>>, tensor<32x3x3x3x1x!quant.uniform<i8:f32, 0.05:0>>, none) -> tensor<1x14x30x30x32x!quant.uniform<i8:f32, 0.1:0>>
    
    return %conv : tensor<1x14x30x30x32x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: kernel_shape = [3, 3, 3]

// -----

// =============================================================================
// Test 8: Depthwise conv with 5x5 kernel - SHOULD CONVERT
// =============================================================================
// CHECK-LABEL: @depthwise_conv2d_5x5_kernel
func.func @depthwise_conv2d_5x5_kernel(%arg0: tensor<1x28x28x96x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x28x28x96x!quant.uniform<i8:f32, 0.1:0>> {
    // Weight in OHWI format: [C_out=96, kH=5, kW=5, C_in/group=1]
    %weights = onnx.Constant {value = dense<1> : tensor<96x5x5x1xi8>} : tensor<96x5x5x1x!quant.uniform<i8:f32, 0.05:0>>
    %none = "onnx.NoValue"() {value} : () -> none
    
    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 96 : si64,
        kernel_shape = [5, 5],
        pads = [2, 2, 2, 2],
        strides = [1, 1]
    } : (tensor<1x28x28x96x!quant.uniform<i8:f32, 0.1:0>>, tensor<96x5x5x1x!quant.uniform<i8:f32, 0.05:0>>, none) -> tensor<1x28x28x96x!quant.uniform<i8:f32, 0.1:0>>
    
    return %conv : tensor<1x28x28x96x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: kernel_shape = [5, 5]

// -----

// =============================================================================
// Test 9: Multiple depthwise convs in sequence - SHOULD CONVERT BOTH
// =============================================================================
// CHECK-LABEL: @multiple_depthwise_convs
func.func @multiple_depthwise_convs(%arg0: tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>> {
    // Weight in OHWI format: [C_out=64, kH=3, kW=3, C_in/group=1]
    %weights1 = onnx.Constant {value = dense<1> : tensor<64x3x3x1xi8>} : tensor<64x3x3x1x!quant.uniform<i8:f32, 0.05:0>>
    %weights2 = onnx.Constant {value = dense<1> : tensor<64x3x3x1xi8>} : tensor<64x3x3x1x!quant.uniform<i8:f32, 0.05:0>>
    %none = "onnx.NoValue"() {value} : () -> none
    
    // First depthwise conv
    %conv1 = "onnx.XFEConv"(%arg0, %weights1, %none) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 64 : si64,
        kernel_shape = [3, 3],
        pads = [1, 1, 1, 1],
        strides = [1, 1]
    } : (tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>, tensor<64x3x3x1x!quant.uniform<i8:f32, 0.05:0>>, none) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>
    
    // Second depthwise conv
    %conv2 = "onnx.XFEConv"(%conv1, %weights2, %none) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 64 : si64,
        kernel_shape = [3, 3],
        pads = [1, 1, 1, 1],
        strides = [1, 1]
    } : (tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>, tensor<64x3x3x1x!quant.uniform<i8:f32, 0.05:0>>, none) -> tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>
    
    return %conv2 : tensor<1x56x56x64x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK: onnx.XCOMPILERDepthwiseConv

// -----

// =============================================================================
// Test 10: Float 2D depthwise conv (f32) - SHOULD CONVERT
// =============================================================================
// CHECK-LABEL: @depthwise_conv2d_f32
func.func @depthwise_conv2d_f32(%arg0: tensor<1x56x56x64xf32>) -> tensor<1x54x54x64xf32> {
    // Weight in OHWI format: [C_out=64, kH=3, kW=3, C_in/group=1]
    %weights = onnx.Constant {value = dense<1.0> : tensor<64x3x3x1xf32>} : tensor<64x3x3x1xf32>
    %none = "onnx.NoValue"() {value} : () -> none

    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "NOTSET",
        dilations = [1, 1],
        group = 64 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    } : (tensor<1x56x56x64xf32>, tensor<64x3x3x1xf32>, none) -> tensor<1x54x54x64xf32>

    return %conv : tensor<1x54x54x64xf32>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: kernel_shape = [3, 3]

// -----

// =============================================================================
// Test 11: Float 2D depthwise conv with bias (f32) - SHOULD CONVERT
// =============================================================================
// CHECK-LABEL: @depthwise_conv2d_f32_with_bias
func.func @depthwise_conv2d_f32_with_bias(%arg0: tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32> {
    // Weight in OHWI format: [C_out=32, kH=3, kW=3, C_in/group=1]
    %weights = onnx.Constant {value = dense<1.0> : tensor<32x3x3x1xf32>} : tensor<32x3x3x1xf32>
    %bias = onnx.Constant {value = dense<0.0> : tensor<32xf32>} : tensor<32xf32>

    %conv = "onnx.XFEConv"(%arg0, %weights, %bias) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 32 : si64,
        kernel_shape = [3, 3],
        pads = [1, 1, 1, 1],
        strides = [1, 1]
    } : (tensor<1x28x28x32xf32>, tensor<32x3x3x1xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>

    return %conv : tensor<1x28x28x32xf32>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: auto_pad = "SAME_UPPER"

// -----

// =============================================================================
// Test 12: Float 2D depthwise conv with strides (f16) - SHOULD CONVERT
// =============================================================================
// CHECK-LABEL: @depthwise_conv2d_f16_strided
func.func @depthwise_conv2d_f16_strided(%arg0: tensor<1x112x112x64xf16>) -> tensor<1x56x56x64xf16> {
    // Weight in OHWI format: [C_out=64, kH=3, kW=3, C_in/group=1]
    %weights = onnx.Constant {value = dense<1.0> : tensor<64x3x3x1xf16>} : tensor<64x3x3x1xf16>
    %none = "onnx.NoValue"() {value} : () -> none

    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 64 : si64,
        kernel_shape = [3, 3],
        pads = [0, 0, 1, 1],
        strides = [2, 2]
    } : (tensor<1x112x112x64xf16>, tensor<64x3x3x1xf16>, none) -> tensor<1x56x56x64xf16>

    return %conv : tensor<1x56x56x64xf16>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: strides = [2, 2]

// -----

// =============================================================================
// Test 13: Float 3D depthwise conv (f32) - SHOULD CONVERT
// =============================================================================
// CHECK-LABEL: @depthwise_conv3d_f32
func.func @depthwise_conv3d_f32(%arg0: tensor<1x16x32x32x32xf32>) -> tensor<1x14x30x30x32xf32> {
    // Weight in ODHWI format: [C_out=32, kD=3, kH=3, kW=3, C_in/group=1]
    %weights = onnx.Constant {value = dense<1.0> : tensor<32x3x3x3x1xf32>} : tensor<32x3x3x3x1xf32>
    %none = "onnx.NoValue"() {value} : () -> none

    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "NOTSET",
        dilations = [1, 1, 1],
        group = 32 : si64,
        kernel_shape = [3, 3, 3],
        pads = [0, 0, 0, 0, 0, 0],
        strides = [1, 1, 1]
    } : (tensor<1x16x32x32x32xf32>, tensor<32x3x3x3x1xf32>, none) -> tensor<1x14x30x30x32xf32>

    return %conv : tensor<1x14x30x30x32xf32>
}
// CHECK-NOT: onnx.XFEConv
// CHECK: onnx.XCOMPILERDepthwiseConv
// CHECK-SAME: kernel_shape = [3, 3, 3]

// -----

// =============================================================================
// Test 14: Float regular conv (group = 1, f32) - SHOULD NOT CONVERT
// =============================================================================
// CHECK-LABEL: @regular_conv_f32_not_depthwise
func.func @regular_conv_f32_not_depthwise(%arg0: tensor<1x56x56x3xf32>) -> tensor<1x56x56x64xf32> {
    // Weight in OHWI format: [C_out=64, kH=3, kW=3, C_in/group=3]
    %weights = onnx.Constant {value = dense<1.0> : tensor<64x3x3x3xf32>} : tensor<64x3x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none

    %conv = "onnx.XFEConv"(%arg0, %weights, %none) {
        auto_pad = "SAME_UPPER",
        dilations = [1, 1],
        group = 1 : si64,
        kernel_shape = [3, 3],
        pads = [1, 1, 1, 1],
        strides = [1, 1]
    } : (tensor<1x56x56x3xf32>, tensor<64x3x3x3xf32>, none) -> tensor<1x56x56x64xf32>

    return %conv : tensor<1x56x56x64xf32>
}
// CHECK-NOT: onnx.XCOMPILERDepthwiseConv
// CHECK: onnx.XFEConv
