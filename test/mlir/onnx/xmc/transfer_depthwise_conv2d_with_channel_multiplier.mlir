// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --transfer-depthwise-conv2d-with-channel-multiplier %s | FileCheck %s

module {
  // =========================================================================
  // Positive tests - These SHOULD be converted (depthwise with channel_multiplier > 1)
  // =========================================================================

  // Test 1: Basic depthwise conv with channel_multiplier=2
  // Input: [1, 4, 8, 8] (NCHW), group=4, output_channels=8
  // channel_multiplier = 8/4 = 2
  // Should be split into 2 depthwise convs + concat
  func.func @test_depthwise_conv_cm2(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x8x8x8xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<8x1x3x3xf32>} : () -> tensor<8x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x4x8x8xf32>, tensor<8x1x3x3xf32>, none) -> tensor<1x8x8x8xf32>
    return %0 : tensor<1x8x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_cm2
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.Conv
  // CHECK-SAME: group = 4
  // CHECK: onnx.Conv
  // CHECK-SAME: group = 4
  // CHECK: onnx.Concat
  // CHECK-SAME: axis = 1

  // Test 2: Depthwise conv with channel_multiplier=3
  // Input: [1, 2, 8, 8], group=2, output_channels=6
  // channel_multiplier = 6/2 = 3
  func.func @test_depthwise_conv_cm3(%arg0: tensor<1x2x8x8xf32>) -> tensor<1x6x8x8xf32> {
    %weights = "onnx.Constant"() {value = dense<0.5> : tensor<6x1x3x3xf32>} : () -> tensor<6x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 2 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x2x8x8xf32>, tensor<6x1x3x3xf32>, none) -> tensor<1x6x8x8xf32>
    return %0 : tensor<1x6x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_cm3
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat
  // CHECK-SAME: axis = 1

  // Test 3: Depthwise conv with channel_multiplier=4
  // Input: [1, 3, 16, 16], group=3, output_channels=12
  // channel_multiplier = 12/3 = 4
  func.func @test_depthwise_conv_cm4(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x12x14x14xf32> {
    %weights = "onnx.Constant"() {value = dense<0.25> : tensor<12x1x3x3xf32>} : () -> tensor<12x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 3 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
    } : (tensor<1x3x16x16xf32>, tensor<12x1x3x3xf32>, none) -> tensor<1x12x14x14xf32>
    return %0 : tensor<1x12x14x14xf32>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_cm4
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat

  // Test 4: Depthwise conv with bias and channel_multiplier=2
  func.func @test_depthwise_conv_with_bias(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x8x8x8xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<8x1x3x3xf32>} : () -> tensor<8x1x3x3xf32>
    %bias = "onnx.Constant"() {value = dense<0.1> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "onnx.Conv"(%arg0, %weights, %bias) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x4x8x8xf32>, tensor<8x1x3x3xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
    return %0 : tensor<1x8x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_with_bias
  // CHECK: onnx.Constant
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat

  // Test 5: Depthwise conv with 1x1 kernel and channel_multiplier=2
  func.func @test_depthwise_conv_1x1_cm2(%arg0: tensor<1x8x16x16xf32>) -> tensor<1x16x16x16xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<16x1x1x1xf32>} : () -> tensor<16x1x1x1xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 8 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
    } : (tensor<1x8x16x16xf32>, tensor<16x1x1x1xf32>, none) -> tensor<1x16x16x16xf32>
    return %0 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_1x1_cm2
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat

  // Test 6: Depthwise conv with stride and channel_multiplier=2
  func.func @test_depthwise_conv_stride(%arg0: tensor<1x4x16x16xf32>) -> tensor<1x8x8x8xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<8x1x3x3xf32>} : () -> tensor<8x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [2, 2]
    } : (tensor<1x4x16x16xf32>, tensor<8x1x3x3xf32>, none) -> tensor<1x8x8x8xf32>
    return %0 : tensor<1x8x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_stride
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat

  // =========================================================================
  // Negative tests - These should NOT be converted
  // =========================================================================

  // Test 7: Depthwise conv with channel_multiplier=1 (should NOT be split)
  func.func @test_depthwise_conv_cm1(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<4x1x3x3xf32>} : () -> tensor<4x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x4x8x8xf32>, tensor<4x1x3x3xf32>, none) -> tensor<1x4x8x8xf32>
    return %0 : tensor<1x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_cm1
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.Conv
  // CHECK-NOT: onnx.Concat

  // Test 8: Regular conv (not depthwise) - group=1 (should NOT be split)
  func.func @test_regular_conv(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x8x8x8xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<8x4x3x3xf32>} : () -> tensor<8x4x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x8x8xf32>
    return %0 : tensor<1x8x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_regular_conv
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.Conv
  // CHECK-NOT: onnx.Concat

  // Test 9: Group conv (not depthwise) - group != input_channels (should NOT be split)
  func.func @test_group_conv(%arg0: tensor<1x8x8x8xf32>) -> tensor<1x16x8x8xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<16x2x3x3xf32>} : () -> tensor<16x2x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x8x8x8xf32>, tensor<16x2x3x3xf32>, none) -> tensor<1x16x8x8xf32>
    return %0 : tensor<1x16x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_group_conv
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.Conv
  // CHECK-NOT: onnx.Concat

  // =========================================================================
  // XFE Conv Tests (channel-last layout)
  // =========================================================================

  // Test 10: XFE depthwise conv with channel_multiplier=2 (NHWC layout)
  // Input: [1, 8, 8, 4] (NHWC), group=4, output_channels=8
  func.func @test_xfe_depthwise_conv_cm2(%arg0: tensor<1x8x8x4xf32>) -> tensor<1x8x8x8xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<8x1x3x3xf32>} : () -> tensor<8x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.XFEConv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x8x8x4xf32>, tensor<8x1x3x3xf32>, none) -> tensor<1x8x8x8xf32>
    return %0 : tensor<1x8x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_xfe_depthwise_conv_cm2
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.XFEConv
  // CHECK: onnx.XFEConv
  // CHECK: onnx.Concat
  // CHECK-SAME: axis = 3

  // Test 11: XFE depthwise conv with channel_multiplier=1 (should NOT be split)
  func.func @test_xfe_depthwise_conv_cm1(%arg0: tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<4x1x3x3xf32>} : () -> tensor<4x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.XFEConv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x8x8x4xf32>, tensor<4x1x3x3xf32>, none) -> tensor<1x8x8x4xf32>
    return %0 : tensor<1x8x8x4xf32>
  }
  // CHECK-LABEL: func.func @test_xfe_depthwise_conv_cm1
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.XFEConv
  // CHECK-NOT: onnx.Concat

  // =========================================================================
  // Integration tests - depthwise conv with other operations
  // =========================================================================

  // Test 12: Depthwise conv followed by ReLU
  func.func @test_depthwise_conv_relu(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x8x8x8xf32> {
    %weights = "onnx.Constant"() {value = dense<1.0> : tensor<8x1x3x3xf32>} : () -> tensor<8x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x4x8x8xf32>, tensor<8x1x3x3xf32>, none) -> tensor<1x8x8x8xf32>
    %1 = "onnx.Relu"(%0) : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
    return %1 : tensor<1x8x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_relu
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat
  // CHECK: onnx.Relu

  // Test 13: Depthwise conv followed by regular conv (common pattern)
  func.func @test_depthwise_followed_by_conv(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x16x8x8xf32> {
    %weights1 = "onnx.Constant"() {value = dense<1.0> : tensor<8x1x3x3xf32>} : () -> tensor<8x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights1, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x4x8x8xf32>, tensor<8x1x3x3xf32>, none) -> tensor<1x8x8x8xf32>
    %weights2 = "onnx.Constant"() {value = dense<0.5> : tensor<16x8x1x1xf32>} : () -> tensor<16x8x1x1xf32>
    %1 = "onnx.Conv"(%0, %weights2, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 1 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
    } : (tensor<1x8x8x8xf32>, tensor<16x8x1x1xf32>, none) -> tensor<1x16x8x8xf32>
    return %1 : tensor<1x16x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_depthwise_followed_by_conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat
  // The second conv (regular conv) should NOT have a concat after it
  // CHECK: onnx.Conv
  // CHECK-NEXT: return

  // Test 14: Multiple depthwise convs in sequence
  func.func @test_multiple_depthwise_convs(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x16x8x8xf32> {
    %weights1 = "onnx.Constant"() {value = dense<1.0> : tensor<8x1x3x3xf32>} : () -> tensor<8x1x3x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights1, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x4x8x8xf32>, tensor<8x1x3x3xf32>, none) -> tensor<1x8x8x8xf32>
    %weights2 = "onnx.Constant"() {value = dense<0.5> : tensor<16x1x3x3xf32>} : () -> tensor<16x1x3x3xf32>
    %1 = "onnx.Conv"(%0, %weights2, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 8 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x8x8x8xf32>, tensor<16x1x3x3xf32>, none) -> tensor<1x16x8x8xf32>
    return %1 : tensor<1x16x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_multiple_depthwise_convs
  // Both depthwise convs should be split
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat

  // =========================================================================
  // Quantized type tests - Verify storage vs quantized type handling
  // The fix ensures split weight/bias constants use the full quantized type
  // (e.g., !quant.uniform<i8:f32, ...>) for the result, not just the storage
  // type (e.g., i8).
  // =========================================================================

  // Test 15: Quantized depthwise conv with channel_multiplier=2 (NCHW, ui8)
  // Input: [1, 4, 8, 8] (NCHW), group=4, output_channels=8
  // channel_multiplier = 8/4 = 2
  // Verifies that split weight constants preserve the full quantized type
  // (e.g., !quant.uniform<u8:f32, ...>) instead of using just the storage type (ui8).
  func.func @test_depthwise_conv_cm2_quant_ui8(%arg0: tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>> {
    %weights = "onnx.Constant"() {value = dense<128> : tensor<8x1x3x3xui8>} : () -> tensor<8x1x3x3x!quant.uniform<u8:f32, 0.05:128>>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>, tensor<8x1x3x3x!quant.uniform<u8:f32, 0.05:128>>, none) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>
    return %0 : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_cm2_quant_ui8
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.Conv
  // CHECK-SAME: group = 4
  // CHECK: onnx.Conv
  // CHECK-SAME: group = 4
  // CHECK: onnx.Concat
  // CHECK-SAME: axis = 1

  // Test 16: Quantized depthwise conv with bias and channel_multiplier=2 (NCHW, ui8)
  // Bias uses quantized i32 type, verifying both weight and bias type handling
  func.func @test_depthwise_conv_cm2_quant_ui8_bias(%arg0: tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>> {
    %weights = "onnx.Constant"() {value = dense<128> : tensor<8x1x3x3xui8>} : () -> tensor<8x1x3x3x!quant.uniform<u8:f32, 0.05:128>>
    %bias = "onnx.Constant"() {value = dense<0> : tensor<8xi32>} : () -> tensor<8x!quant.uniform<i32:f32, 0.005:0>>
    %0 = "onnx.Conv"(%arg0, %weights, %bias) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>, tensor<8x1x3x3x!quant.uniform<u8:f32, 0.05:128>>, tensor<8x!quant.uniform<i32:f32, 0.005:0>>) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>
    return %0 : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>
  }
  // CHECK-LABEL: func.func @test_depthwise_conv_cm2_quant_ui8_bias
  // CHECK: onnx.Constant
  // CHECK: onnx.Conv
  // CHECK: onnx.Conv
  // CHECK: onnx.Concat
  // CHECK-SAME: axis = 1

  // Test 17: XFE quantized depthwise conv with channel_multiplier=2 (NHWC, ui8)
  // Input: [1, 8, 8, 4] (NHWC), group=4, output_channels=8
  // Verifies that XFEConv split preserves quantized types for weight constants.
  func.func @test_xfe_depthwise_conv_cm2_quant_ui8(%arg0: tensor<1x8x8x4x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>> {
    %weights = "onnx.Constant"() {value = dense<128> : tensor<8x1x3x3xui8>} : () -> tensor<8x1x3x3x!quant.uniform<u8:f32, 0.05:128>>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.XFEConv"(%arg0, %weights, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 4 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x8x8x4x!quant.uniform<u8:f32, 0.1:128>>, tensor<8x1x3x3x!quant.uniform<u8:f32, 0.05:128>>, none) -> tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>
    return %0 : tensor<1x8x8x8x!quant.uniform<u8:f32, 0.1:128>>
  }
  // CHECK-LABEL: func.func @test_xfe_depthwise_conv_cm2_quant_ui8
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.XFEConv
  // CHECK: onnx.XFEConv
  // CHECK: onnx.Concat
  // CHECK-SAME: axis = 3

  // Test 18: XFE quantized depthwise conv with bias and channel_multiplier=3 (NHWC, ui8)
  func.func @test_xfe_depthwise_conv_cm3_quant_bias(%arg0: tensor<1x8x8x2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x8x8x6x!quant.uniform<u8:f32, 0.1:128>> {
    %weights = "onnx.Constant"() {value = dense<128> : tensor<6x1x3x3xui8>} : () -> tensor<6x1x3x3x!quant.uniform<u8:f32, 0.05:128>>
    %bias = "onnx.Constant"() {value = dense<0> : tensor<6xi32>} : () -> tensor<6x!quant.uniform<i32:f32, 0.005:0>>
    %0 = "onnx.XFEConv"(%arg0, %weights, %bias) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 2 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x8x8x2x!quant.uniform<u8:f32, 0.1:128>>, tensor<6x1x3x3x!quant.uniform<u8:f32, 0.05:128>>, tensor<6x!quant.uniform<i32:f32, 0.005:0>>) -> tensor<1x8x8x6x!quant.uniform<u8:f32, 0.1:128>>
    return %0 : tensor<1x8x8x6x!quant.uniform<u8:f32, 0.1:128>>
  }
  // CHECK-LABEL: func.func @test_xfe_depthwise_conv_cm3_quant_bias
  // CHECK: onnx.Constant
  // CHECK: onnx.XFEConv
  // CHECK: onnx.XFEConv
  // CHECK: onnx.XFEConv
  // CHECK: onnx.Concat
  // CHECK-SAME: axis = 3

  // Test 19: Per-axis quantized depthwise conv with channel_multiplier=2 (NCHW)
  // 2 input channels, 4 output channels, cm=2. Per-axis scales on axis 0.
  // Split into 2 convs: channels [0,1] and [2,3].
  // Scales {0.1, 0.2, 0.3, 0.4} -> {0.1, 0.2} and {0.3, 0.4}
  func.func @test_depthwise_per_axis_cm2(%arg0: tensor<1x2x8x8x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>> {
    %weights = "onnx.Constant"() {value = dense<128> : tensor<4x1x3x3xui8>} : () -> tensor<4x1x3x3x!quant.uniform<u8:f32:0, {0.1, 0.2, 0.3, 0.4}>>
    %bias = "onnx.Constant"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4x!quant.uniform<i32:f32:0, {0.01, 0.02, 0.03, 0.04}>>
    %0 = "onnx.Conv"(%arg0, %weights, %bias) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 2 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
    } : (tensor<1x2x8x8x!quant.uniform<u8:f32, 0.1:128>>, tensor<4x1x3x3x!quant.uniform<u8:f32:0, {0.1, 0.2, 0.3, 0.4}>>, tensor<4x!quant.uniform<i32:f32:0, {0.01, 0.02, 0.03, 0.04}>>) -> tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>
    return %0 : tensor<1x4x8x8x!quant.uniform<u8:f32, 0.1:128>>
  }
  // CHECK-LABEL: func.func @test_depthwise_per_axis_cm2
  // First split: channels [0,1], scales {0.1, 0.2}, bias scales {0.01, 0.02}
  // CHECK: onnx.Conv
  // CHECK-SAME: tensor<2x1x3x3x!quant.uniform<u8:f32:0, {1.000000e-01,2.000000e-01}>>
  // CHECK-SAME: tensor<2x!quant.uniform<i32:f32:0, {1.000000e-02,2.000000e-02}>>
  // Second split: channels [2,3], scales {0.3, 0.4}, bias scales {0.03, 0.04}
  // CHECK: onnx.Conv
  // CHECK-SAME: tensor<2x1x3x3x!quant.uniform<u8:f32:0, {3.000000e-01,4.000000e-01}>>
  // CHECK-SAME: tensor<2x!quant.uniform<i32:f32:0, {3.000000e-02,4.000000e-02}>>
  // CHECK: onnx.Concat
}
