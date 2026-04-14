// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --transfer-op-3d-to-op-2d %s | FileCheck %s

module {
  // Test 1: Convert Conv3d to Conv2d
  // Input: 5D Conv3d [N, C, D, H, W] (ONNX channel-first format) with 5D weights
  // Expected: Reshape -> Conv2d -> Reshape back to 5D
  func.func @test_conv3d(%arg0: tensor<1x16x4x8x8xf32>, %arg1: tensor<32x16x3x3x3xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x4x8x8xf32>, tensor<32x16x3x3x3xf32>, none) -> tensor<1x32x4x8x8xf32>

    return %1 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 2: Convert matmul-like Conv3d (all 1s for kernel, stride, padding, dilation)
  // This should be handled with special reshaping strategy
  func.func @test_matmul_like_conv3d(%arg0: tensor<1x16x4x8x8xf32>, %arg1: tensor<32x16x1x1x1xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [1, 1, 1],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x4x8x8xf32>, tensor<32x16x1x1x1xf32>, none) -> tensor<1x32x4x8x8xf32>

    return %1 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_matmul_like_conv3d
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 3: Convert Conv3d + Relu fusion
  // Should transform both operations together
  func.func @test_conv3d_relu(%arg0: tensor<1x16x4x8x8xf32>, %arg1: tensor<32x16x3x3x3xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x4x8x8xf32>, tensor<32x16x3x3x3xf32>, none) -> tensor<1x32x4x8x8xf32>

    %2 = "onnx.Relu"(%1) : (tensor<1x32x4x8x8xf32>) -> tensor<1x32x4x8x8xf32>

    return %2 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d_relu
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Relu
  // CHECK: onnx.Reshape

  // Test 4: Convert 5D element-wise Add to 4D
  // Input: 5D tensors [N, C, D, H, W] (ONNX channel-first format)
  // Expected: Reshape both inputs -> Add -> Reshape back to 5D
  func.func @test_eltwise_add_3d(%arg0: tensor<1x16x4x8x8xf32>, %arg1: tensor<1x16x4x8x8xf32>) -> tensor<1x16x4x8x8xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16x4x8x8xf32>, tensor<1x16x4x8x8xf32>) -> tensor<1x16x4x8x8xf32>

    return %0 : tensor<1x16x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_eltwise_add_3d
  // CHECK: onnx.Reshape
  // CHECK: onnx.Reshape
  // CHECK: onnx.Add
  // CHECK: onnx.Reshape

  // Test 5: Should NOT match - grouped convolution (group != 1)
  // This should remain unchanged
  func.func @test_grouped_conv3d(%arg0: tensor<1x16x4x8x8xf32>, %arg1: tensor<32x8x3x3x3xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 2 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x4x8x8xf32>, tensor<32x8x3x3x3xf32>, none) -> tensor<1x32x4x8x8xf32>

    return %1 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_grouped_conv3d
  // CHECK-NOT: onnx.Reshape
  // CHECK: onnx.Conv

  // Test 6: Should NOT match - 4D convolution (already 2D)
  // This should remain unchanged
  func.func @test_conv2d(%arg0: tensor<1x16x8x8xf32>, %arg1: tensor<32x16x3x3xf32>) -> tensor<1x32x6x6xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
    } : (tensor<1x16x8x8xf32>, tensor<32x16x3x3xf32>, none) -> tensor<1x32x6x6xf32>

    return %1 : tensor<1x32x6x6xf32>
  }
  // CHECK-LABEL: func.func @test_conv2d
  // CHECK-NOT: onnx.Reshape
  // CHECK: onnx.Conv

  // Test 7: Conv3D with bias (no ReLU)
  // Should add bias to the Conv2d operation
  func.func @test_conv3d_with_bias(%arg0: tensor<1x16x4x8x8xf32>, %arg1: tensor<32x16x3x3x3xf32>, %arg2: tensor<32xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x4x8x8xf32>, tensor<32x16x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x4x8x8xf32>

    return %0 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d_with_bias
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 8: Conv3D with bias and ReLU
  // Should transform both operations with bias support
  func.func @test_conv3d_bias_relu(%arg0: tensor<1x16x4x8x8xf32>, %arg1: tensor<32x16x3x3x3xf32>, %arg2: tensor<32xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x4x8x8xf32>, tensor<32x16x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x4x8x8xf32>

    %1 = "onnx.Relu"(%0) : (tensor<1x32x4x8x8xf32>) -> tensor<1x32x4x8x8xf32>

    return %1 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d_bias_relu
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Relu
  // CHECK: onnx.Reshape

  // Test 9: Eltwise3D Add with ReLU
  // Should transform both add and relu operations
  func.func @test_eltwise_add_3d_relu(%arg0: tensor<1x16x4x8x8xf32>, %arg1: tensor<1x16x4x8x8xf32>) -> tensor<1x16x4x8x8xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16x4x8x8xf32>, tensor<1x16x4x8x8xf32>) -> tensor<1x16x4x8x8xf32>
    %1 = "onnx.Relu"(%0) : (tensor<1x16x4x8x8xf32>) -> tensor<1x16x4x8x8xf32>

    return %1 : tensor<1x16x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_eltwise_add_3d_relu
  // CHECK: onnx.Reshape
  // CHECK: onnx.Reshape
  // CHECK: onnx.Add
  // CHECK: onnx.Relu
  // CHECK: onnx.Reshape

  // Test 10: Conv3D with single channel input (edge case)
  // Should handle single channel inputs correctly
  func.func @test_conv3d_single_channel(%arg0: tensor<1x1x4x8x8xf32>, %arg1: tensor<32x1x3x3x3xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x1x4x8x8xf32>, tensor<32x1x3x3x3xf32>, none) -> tensor<1x32x4x8x8xf32>

    return %1 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d_single_channel
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 11: Conv3D with large depth dimension
  // Tests behavior with larger depth values
  func.func @test_conv3d_large_depth(%arg0: tensor<1x16x16x8x8xf32>, %arg1: tensor<32x16x3x3x3xf32>) -> tensor<1x32x16x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x16x8x8xf32>, tensor<32x16x3x3x3xf32>, none) -> tensor<1x32x16x8x8xf32>

    return %1 : tensor<1x32x16x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d_large_depth
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 12: Conv3D with small depth dimension
  // Tests behavior with minimal depth
  func.func @test_conv3d_small_depth(%arg0: tensor<1x16x2x8x8xf32>, %arg1: tensor<32x16x2x3x3xf32>) -> tensor<1x32x2x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [2, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x2x8x8xf32>, tensor<32x16x2x3x3xf32>, none) -> tensor<1x32x2x8x8xf32>

    return %1 : tensor<1x32x2x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d_small_depth
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 13: Conv3D with stride in depth dimension
  // Tests strided convolution along depth
  func.func @test_conv3d_strided(%arg0: tensor<1x16x8x8x8xf32>, %arg1: tensor<32x16x3x3x3xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [2, 1, 1]
    } : (tensor<1x16x8x8x8xf32>, tensor<32x16x3x3x3xf32>, none) -> tensor<1x32x4x8x8xf32>

    return %1 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d_strided
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 14: Conv3D with padding in depth dimension
  // Tests padded convolution along depth
  func.func @test_conv3d_padded(%arg0: tensor<1x16x4x8x8xf32>, %arg1: tensor<32x16x3x3x3xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [1, 0, 0, 1, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x4x8x8xf32>, tensor<32x16x3x3x3xf32>, none) -> tensor<1x32x4x8x8xf32>

    return %1 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d_padded
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 15: Conv3D with dilation in depth dimension
  // Tests dilated convolution along depth
  func.func @test_conv3d_dilated(%arg0: tensor<1x16x8x8x8xf32>, %arg1: tensor<32x16x3x3x3xf32>) -> tensor<1x32x4x8x8xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none

    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [2, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x16x8x8x8xf32>, tensor<32x16x3x3x3xf32>, none) -> tensor<1x32x4x8x8xf32>

    return %1 : tensor<1x32x4x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_conv3d_dilated
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 16: Eltwise3D Add with detailed dimension transformation
  // Demonstrates the 5D to 4D transformation for element-wise operations
  //
  // ONNX Format (Channel-first):
  //   Original 5D: [N, C, D, H, W] = [2, 8, 4, 16, 16]
  //   Target 4D:   [N, C*D, H, W] = [2, 32, 16, 16]
  //
  // XIR Format (Channel-last - for reference):
  //   Original 5D: [N, H, W, D, C] = [2, 16, 16, 4, 8]
  //   Target 4D:   [N, H, W, D*C] = [2, 16, 16, 32]
  //
  // This test validates:
  // - Depth dimension (D=4) is merged with channels (C=8) → C*D=32
  // - Both inputs reshaped independently
  // - Element-wise operation performed on 4D
  // - Output reshaped back to 5D
  func.func @test_eltwise_add_3d_detailed(%arg0: tensor<2x8x4x16x16xf32>, %arg1: tensor<2x8x4x16x16xf32>) -> tensor<2x8x4x16x16xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x8x4x16x16xf32>, tensor<2x8x4x16x16xf32>) -> tensor<2x8x4x16x16xf32>

    return %0 : tensor<2x8x4x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_eltwise_add_3d_detailed
  // CHECK: onnx.Constant{{.*}}tensor<5xi64>
  // CHECK: onnx.Constant{{.*}}tensor<4xi64>
  // CHECK: onnx.Reshape{{.*}}(tensor<2x8x4x16x16xf32>{{.*}}) -> tensor<2x32x16x16xf32>
  // CHECK: onnx.Reshape{{.*}}(tensor<2x8x4x16x16xf32>{{.*}}) -> tensor<2x32x16x16xf32>
  // CHECK: onnx.Add{{.*}}(tensor<2x32x16x16xf32>, tensor<2x32x16x16xf32>) -> tensor<2x32x16x16xf32>
  // CHECK: onnx.Reshape{{.*}}(tensor<2x32x16x16xf32>{{.*}}) -> tensor<2x8x4x16x16xf32>

  // Test 17: Eltwise3D with multiple channels and depth
  // Tests dimension merging with larger values
  // C*D = 16*8 = 128 merged channels
  func.func @test_eltwise_large_merge(%arg0: tensor<1x16x8x32x32xf32>, %arg1: tensor<1x16x8x32x32xf32>) -> tensor<1x16x8x32x32xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16x8x32x32xf32>, tensor<1x16x8x32x32xf32>) -> tensor<1x16x8x32x32xf32>

    return %0 : tensor<1x16x8x32x32xf32>
  }
  // CHECK-LABEL: func.func @test_eltwise_large_merge
  // CHECK: onnx.Constant{{.*}}tensor<5xi64>
  // CHECK: onnx.Constant{{.*}}tensor<4xi64>
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x128x32x32xf32>
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x128x32x32xf32>
  // CHECK: onnx.Add{{.*}}(tensor<1x128x32x32xf32>, tensor<1x128x32x32xf32>)
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x8x32x32xf32>

  // Test 18: Eltwise3D minimal dimensions
  // Edge case: minimal channel and depth dimensions
  // C*D = 1*2 = 2 merged channels
  func.func @test_eltwise_minimal(%arg0: tensor<1x1x2x8x8xf32>, %arg1: tensor<1x1x2x8x8xf32>) -> tensor<1x1x2x8x8xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x1x2x8x8xf32>, tensor<1x1x2x8x8xf32>) -> tensor<1x1x2x8x8xf32>

    return %0 : tensor<1x1x2x8x8xf32>
  }
  // CHECK-LABEL: func.func @test_eltwise_minimal
  // CHECK: onnx.Constant{{.*}}tensor<5xi64>
  // CHECK: onnx.Constant{{.*}}tensor<4xi64>
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x2x8x8xf32>
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x2x8x8xf32>
  // CHECK: onnx.Add{{.*}}(tensor<1x2x8x8xf32>, tensor<1x2x8x8xf32>)
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x1x2x8x8xf32>

  // Test 19: Per-axis quantized matmul-like Conv3d → Conv2d
  // Matmul-like: kernel=1, stride=1, pad=0, dilation=1
  // Weight: [OC=2, IC=4, 1, 1, 1] → [OC=2, IC*D=16, 1, 1]
  // Axis 0 stays OC=2, per-axis quant unchanged.
  func.func @test_matmul_like_conv3d_per_axis(%arg0: tensor<1x4x4x8x8x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x2x4x8x8x!quant.uniform<i8:f32, 0.1:0>> {
    %w = onnx.Constant {value = dense<1> : tensor<2x4x1x1x1xi8>} : tensor<2x4x1x1x1x!quant.uniform<i8:f32:0, {0.05, 0.06}>>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %w, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [1, 1, 1],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x4x4x8x8x!quant.uniform<i8:f32, 0.1:0>>, tensor<2x4x1x1x1x!quant.uniform<i8:f32:0, {0.05, 0.06}>>, none) -> tensor<1x2x4x8x8x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<1x2x4x8x8x!quant.uniform<i8:f32, 0.1:0>>
  }
  // Matmul-like: OC unchanged at 2, per-axis quant preserved as-is
  // CHECK-LABEL: func.func @test_matmul_like_conv3d_per_axis
  // CHECK: onnx.Reshape
  // CHECK: onnx.Reshape{{.*}} -> tensor<2x16x1x1x!quant.uniform<i8:f32:0, {5.000000e-02,6.000000e-02}>>
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  // Test 20: Per-axis quantized non-matmul Conv3d → Conv2d
  // Non-matmul: kernel=[3,3,3], stride=1, pad=0
  // Weight: [OC=2, IC=2, 3, 3, 3] → [OC*D=8, IC*D=8, 3, 3]
  // Per-axis quant on axis 0: scales {s0, s1} expand to {s0,s0,s0,s0, s1,s1,s1,s1}
  func.func @test_non_matmul_conv3d_per_axis(%arg0: tensor<1x2x4x8x8x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x2x4x8x8x!quant.uniform<i8:f32, 0.1:0>> {
    %w = onnx.Constant {value = dense<1> : tensor<2x2x3x3x3xi8>} : tensor<2x2x3x3x3x!quant.uniform<i8:f32:0, {0.05, 0.06}>>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %w, %none) {
      auto_pad = "NOTSET",
      dilations = [1, 1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3, 3],
      pads = [0, 0, 0, 0, 0, 0],
      strides = [1, 1, 1]
    } : (tensor<1x2x4x8x8x!quant.uniform<i8:f32, 0.1:0>>, tensor<2x2x3x3x3x!quant.uniform<i8:f32:0, {0.05, 0.06}>>, none) -> tensor<1x2x4x8x8x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<1x2x4x8x8x!quant.uniform<i8:f32, 0.1:0>>
  }
  // Non-matmul: OC expands 2→8 (factor D=4), each scale repeated 4 times
  // CHECK-LABEL: func.func @test_non_matmul_conv3d_per_axis
  // CHECK: onnx.Reshape
  // CHECK: onnx.Reshape{{.*}} -> tensor<8x8x3x3x!quant.uniform<i8:f32:0, {5.000000e-02,5.000000e-02,5.000000e-02,5.000000e-02,6.000000e-02,6.000000e-02,6.000000e-02,6.000000e-02}>>
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape
}
