// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --transfer-pool-fix-to-downsample-fix %s | FileCheck %s

module {
  // Test 1: MaxPool with kernel=[1,1] and stride=[2,2] → Resize (nearest)
  // This is the basic case that should be converted
  // Input: [N, C, H, W] = [1, 16, 32, 32]
  // Output: [1, 16, 16, 16] (downsampled by stride)
  func.func @test_maxpool_to_resize_basic(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32>

    return %0 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_to_resize_basic
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.Constant
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // Test 2: AveragePool with kernel=[1,1] and stride=[2,2] → Resize (nearest)
  // AveragePool should also be converted when conditions are met
  func.func @test_avgpool_to_resize_basic(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      strides = [2, 2]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32>

    return %0 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_avgpool_to_resize_basic
  // CHECK-NOT: onnx.AveragePool
  // CHECK: onnx.Constant
  // CHECK: onnx.NoValue
  // CHECK: onnx.Constant
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // Test 3: MaxPool with larger stride values
  // kernel=[1,1], stride=[4,4] → scale = [0.25, 0.25]
  func.func @test_maxpool_stride_4(%arg0: tensor<1x32x64x64xf32>) -> tensor<1x32x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [4, 4]
    } : (tensor<1x32x64x64xf32>) -> tensor<1x32x16x16xf32>

    return %0 : tensor<1x32x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_stride_4
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // Test 4: MaxPool with asymmetric strides
  // kernel=[1,1], stride=[2,4] → different scales for H and W
  func.func @test_maxpool_asymmetric_stride(%arg0: tensor<1x16x32x64xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 4]
    } : (tensor<1x16x32x64xf32>) -> tensor<1x16x16x16xf32>

    return %0 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_asymmetric_stride
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // Test 5: MaxPool with stride=[3,3]
  // Non-power-of-2 stride value
  func.func @test_maxpool_stride_3(%arg0: tensor<1x16x27x27xf32>) -> tensor<1x16x9x9xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [3, 3]
    } : (tensor<1x16x27x27xf32>) -> tensor<1x16x9x9xf32>

    return %0 : tensor<1x16x9x9xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_stride_3
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // =========================================================================
  // Negative tests - These should NOT be converted
  // =========================================================================

  // Test 6: Should NOT match - kernel != [1,1]
  // kernel=[2,2] means actual pooling is performed
  func.func @test_maxpool_kernel_2x2(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [2, 2],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32>

    return %0 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_kernel_2x2
  // CHECK: onnx.MaxPoolSingleOut
  // CHECK-NOT: onnx.Resize

  // Test 7: Should NOT match - stride=[1,1] (not > kernel)
  // No downsampling when stride equals kernel
  func.func @test_maxpool_stride_1(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [1, 1]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32>

    return %0 : tensor<1x16x32x32xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_stride_1
  // CHECK: onnx.MaxPoolSingleOut
  // CHECK-NOT: onnx.Resize

  // Test 8: Should NOT match - asymmetric kernel (only H kernel is 1)
  // kernel=[1,2] doesn't satisfy kernel==1 for both dimensions
  func.func @test_maxpool_asymmetric_kernel(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 2],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32>

    return %0 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_asymmetric_kernel
  // CHECK: onnx.MaxPoolSingleOut
  // CHECK-NOT: onnx.Resize

  // Test 9: Should NOT match - only one dimension has stride > kernel
  // stride=[2,1] means W dimension doesn't satisfy kernel < stride
  func.func @test_maxpool_partial_stride(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x16x32xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 1]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x16x32xf32>

    return %0 : tensor<1x16x16x32xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_partial_stride
  // CHECK: onnx.MaxPoolSingleOut
  // CHECK-NOT: onnx.Resize

  // Test 10: Should NOT match - AveragePool with kernel=[2,2]
  // Same condition applies to AveragePool
  func.func @test_avgpool_kernel_2x2(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [2, 2],
      pads = [0, 0, 0, 0],
      strides = [2, 2]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32>

    return %0 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_avgpool_kernel_2x2
  // CHECK: onnx.AveragePool
  // CHECK-NOT: onnx.Resize

  // Test 11: Should NOT match - has non-zero padding
  // Pool with padding cannot be safely converted to resize
  func.func @test_maxpool_with_padding(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x17x17xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [1, 1, 1, 1],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x17x17xf32>

    return %0 : tensor<1x16x17x17xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_with_padding
  // CHECK: onnx.MaxPoolSingleOut
  // CHECK-NOT: onnx.Resize

  // =========================================================================
  // Additional positive tests with various configurations
  // =========================================================================

  // Test 12: MaxPool with rectangular input
  // Non-square spatial dimensions
  func.func @test_maxpool_rectangular(%arg0: tensor<1x16x64x32xf32>) -> tensor<1x16x32x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x64x32xf32>) -> tensor<1x16x32x16xf32>

    return %0 : tensor<1x16x32x16xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_rectangular
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // Test 13: MaxPool followed by other operations
  // Ensures proper integration in computation graph
  func.func @test_maxpool_with_relu(%arg0: tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32>

    %1 = "onnx.Relu"(%0) : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>

    return %1 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool_with_relu
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"
  // CHECK: onnx.Relu

  // Test 14: Multiple pool operations in sequence
  // Both should be converted independently
  func.func @test_multiple_maxpools(%arg0: tensor<1x16x64x64xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x64x64xf32>) -> tensor<1x16x32x32xf32>

    %1 = "onnx.MaxPoolSingleOut"(%0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32>

    return %1 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_multiple_maxpools
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // Test 15: Mixed pool types - MaxPool and AveragePool
  // Both eligible pools should be converted
  func.func @test_mixed_pools(%arg0: tensor<1x16x64x64xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x64x64xf32>) -> tensor<1x16x32x32xf32>

    %1 = "onnx.AveragePool"(%0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      strides = [2, 2]
    } : (tensor<1x16x32x32xf32>) -> tensor<1x16x16x16xf32>

    return %1 : tensor<1x16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_mixed_pools
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK-NOT: onnx.AveragePool
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // =========================================================================
  // ONNX XFE Op Tests (channel-last layout with transposes)
  // =========================================================================

  // Test 16: XFEAveragePool with kernel=[1,1] and stride=[2,2] → XFEResize
  // This tests the channel-last layout pattern with surrounding transposes
  func.func @test_xfe_avgpool_channel_last(%arg0: tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x28x28x64xf32>
    %1 = "onnx.XFEAveragePool"(%0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      strides = [2, 2]
    } : (tensor<1x28x28x64xf32>) -> tensor<1x14x14x64xf32>
    %2 = "onnx.Transpose"(%1) {perm = [0, 3, 1, 2]} : (tensor<1x14x14x64xf32>) -> tensor<1x64x14x14xf32>
    return %2 : tensor<1x64x14x14xf32>
  }
  // CHECK-LABEL: func.func @test_xfe_avgpool_channel_last
  // CHECK: onnx.Transpose
  // CHECK-NOT: onnx.XFEAveragePool
  // CHECK: onnx.XFEResize
  // CHECK-SAME: mode = "nearest"
  // CHECK: onnx.Transpose

  // Test 17: XFEMaxPool with kernel=[1,1] and stride=[2,2] → XFEResize
  func.func @test_xfe_maxpool_channel_last(%arg0: tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x32x32xf32>) -> tensor<1x32x32x64xf32>
    %1 = "onnx.XFEMaxPool"(%0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x32x32x64xf32>) -> tensor<1x16x16x64xf32>
    %2 = "onnx.Transpose"(%1) {perm = [0, 3, 1, 2]} : (tensor<1x16x16x64xf32>) -> tensor<1x64x16x16xf32>
    return %2 : tensor<1x64x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_xfe_maxpool_channel_last
  // CHECK: onnx.Transpose
  // CHECK-NOT: onnx.XFEMaxPool
  // CHECK: onnx.XFEResize
  // CHECK-SAME: mode = "nearest"
  // CHECK: onnx.Transpose

  // Test 18: XFEAveragePool should NOT match - kernel=[2,2]
  func.func @test_xfe_avgpool_channel_last_no_match(%arg0: tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x28x28x64xf32>
    %1 = "onnx.XFEAveragePool"(%0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [2, 2],
      pads = [0, 0, 0, 0],
      strides = [2, 2]
    } : (tensor<1x28x28x64xf32>) -> tensor<1x14x14x64xf32>
    %2 = "onnx.Transpose"(%1) {perm = [0, 3, 1, 2]} : (tensor<1x14x14x64xf32>) -> tensor<1x64x14x14xf32>
    return %2 : tensor<1x64x14x14xf32>
  }
  // CHECK-LABEL: func.func @test_xfe_avgpool_channel_last_no_match
  // CHECK: onnx.Transpose
  // CHECK: onnx.XFEAveragePool
  // CHECK-NOT: onnx.XFEResize
  // CHECK: onnx.Transpose

  // =========================================================================
  // Quantized Dialect Tests (quant.uniform types)
  // =========================================================================

  // Test 19: Quantized MaxPool with kernel=[1,1] and stride=[2,2] → Resize
  // Uses unsigned 8-bit quantized type (u8)
  func.func @test_maxpool_quant_u8(%arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.0625>>) -> tensor<1x16x16x16x!quant.uniform<u8:f32, 0.0625>> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.0625>>) -> tensor<1x16x16x16x!quant.uniform<u8:f32, 0.0625>>

    return %0 : tensor<1x16x16x16x!quant.uniform<u8:f32, 0.0625>>
  }
  // CHECK-LABEL: func.func @test_maxpool_quant_u8
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // Test 20: Quantized AveragePool with kernel=[1,1] and stride=[2,2] → Resize
  // Uses signed 8-bit quantized type (i8)
  func.func @test_avgpool_quant_i8(%arg0: tensor<1x16x32x32x!quant.uniform<i8:f32, 0.125>>) -> tensor<1x16x16x16x!quant.uniform<i8:f32, 0.125>> {
    %0 = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      strides = [2, 2]
    } : (tensor<1x16x32x32x!quant.uniform<i8:f32, 0.125>>) -> tensor<1x16x16x16x!quant.uniform<i8:f32, 0.125>>

    return %0 : tensor<1x16x16x16x!quant.uniform<i8:f32, 0.125>>
  }
  // CHECK-LABEL: func.func @test_avgpool_quant_i8
  // CHECK-NOT: onnx.AveragePool
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // Test 21: Quantized MaxPool with larger stride=[4,4]
  func.func @test_maxpool_quant_stride4(%arg0: tensor<1x32x64x64x!quant.uniform<u8:f32, 0.03125>>) -> tensor<1x32x16x16x!quant.uniform<u8:f32, 0.03125>> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [4, 4]
    } : (tensor<1x32x64x64x!quant.uniform<u8:f32, 0.03125>>) -> tensor<1x32x16x16x!quant.uniform<u8:f32, 0.03125>>

    return %0 : tensor<1x32x16x16x!quant.uniform<u8:f32, 0.03125>>
  }
  // CHECK-LABEL: func.func @test_maxpool_quant_stride4
  // CHECK-NOT: onnx.MaxPoolSingleOut
  // CHECK: onnx.Resize
  // CHECK-SAME: mode = "nearest"

  // Test 22: Quantized MaxPool should NOT match - kernel=[2,2]
  func.func @test_maxpool_quant_no_match(%arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.0625>>) -> tensor<1x16x16x16x!quant.uniform<u8:f32, 0.0625>> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [2, 2],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.0625>>) -> tensor<1x16x16x16x!quant.uniform<u8:f32, 0.0625>>

    return %0 : tensor<1x16x16x16x!quant.uniform<u8:f32, 0.0625>>
  }
  // CHECK-LABEL: func.func @test_maxpool_quant_no_match
  // CHECK: onnx.MaxPoolSingleOut
  // CHECK-NOT: onnx.Resize

  // Test 23: Quantized XFE MaxPool with kernel=[1,1] and stride=[2,2] → XFE Resize
  // Channel-last layout with quantized types
  func.func @test_xfe_maxpool_quant(%arg0: tensor<1x64x32x32x!quant.uniform<u8:f32, 0.0625>>) -> tensor<1x64x16x16x!quant.uniform<u8:f32, 0.0625>> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x32x32x!quant.uniform<u8:f32, 0.0625>>) -> tensor<1x32x32x64x!quant.uniform<u8:f32, 0.0625>>
    %1 = "onnx.XFEMaxPool"(%0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x32x32x64x!quant.uniform<u8:f32, 0.0625>>) -> tensor<1x16x16x64x!quant.uniform<u8:f32, 0.0625>>
    %2 = "onnx.Transpose"(%1) {perm = [0, 3, 1, 2]} : (tensor<1x16x16x64x!quant.uniform<u8:f32, 0.0625>>) -> tensor<1x64x16x16x!quant.uniform<u8:f32, 0.0625>>
    return %2 : tensor<1x64x16x16x!quant.uniform<u8:f32, 0.0625>>
  }
  // CHECK-LABEL: func.func @test_xfe_maxpool_quant
  // CHECK: onnx.Transpose
  // CHECK-NOT: onnx.XFEMaxPool
  // CHECK: onnx.XFEResize
  // CHECK-SAME: mode = "nearest"
  // CHECK: onnx.Transpose

  // Test 24: Quantized pool with non-zero padding should NOT match
  func.func @test_maxpool_quant_with_padding(%arg0: tensor<1x16x32x32x!quant.uniform<i8:f32, 0.125>>) -> tensor<1x16x17x17x!quant.uniform<i8:f32, 0.125>> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [1, 1],
      pads = [1, 1, 1, 1],
      storage_order = 0 : si64,
      strides = [2, 2]
    } : (tensor<1x16x32x32x!quant.uniform<i8:f32, 0.125>>) -> tensor<1x16x17x17x!quant.uniform<i8:f32, 0.125>>

    return %0 : tensor<1x16x17x17x!quant.uniform<i8:f32, 0.125>>
  }
  // CHECK-LABEL: func.func @test_maxpool_quant_with_padding
  // CHECK: onnx.MaxPoolSingleOut
  // CHECK-NOT: onnx.Resize
}
