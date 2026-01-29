// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt %s -transfer-op1d-to-op2d -o - | FileCheck %s

module {
  //===--------------------------------------------------------------------===//
  // Test 1: Conv1D without activation
  // Input: [N, C, L] = [1, 16, 64] → [N, C, 1, L] = [1, 16, 1, 64]
  // Weight: [OC, IC, K] = [32, 16, 3] → [OC, IC, 1, K] = [32, 16, 1, 3]
  //===--------------------------------------------------------------------===//
  func.func @test_conv1d(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
                         %arg1: tensor<32x16x3x!quant.uniform<i8:f32, 0.05:0>>)
                         -> tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1],
      group = 1 : si64,
      kernel_shape = [3],
      pads = [0, 0],
      strides = [1]
    } : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
         tensor<32x16x3x!quant.uniform<i8:f32, 0.05:0>>, none)
         -> tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>>
    return %1 : tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_conv1d
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x1x64x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.Reshape{{.*}} -> tensor<32x16x1x3x!quant.uniform<i8:f32, 5.000000e-02>>
  // CHECK: onnx.Conv{{.*}}kernel_shape = [1, 3]{{.*}}pads = [0, 0, 0, 0]{{.*}}strides = [1, 1]
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x32x62x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 2: Conv1D with Relu activation
  // Tests fusion of Conv1D + Relu
  //===--------------------------------------------------------------------===//
  func.func @test_conv1d_relu(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
                              %arg1: tensor<32x16x3x!quant.uniform<i8:f32, 0.05:0>>)
                              -> tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1],
      group = 1 : si64,
      kernel_shape = [3],
      pads = [0, 0],
      strides = [1]
    } : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
         tensor<32x16x3x!quant.uniform<i8:f32, 0.05:0>>, none)
         -> tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>>
    %2 = "onnx.Relu"(%1) : (tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>>)
                           -> tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_conv1d_relu
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x1x64x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.Reshape{{.*}} -> tensor<32x16x1x3x!quant.uniform<i8:f32, 5.000000e-02>>
  // CHECK: onnx.Conv
  // CHECK: onnx.Relu
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x32x62x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 3: Conv1D with kernel=1 (special case)
  // When kernel=1 and N=C, input [N, C, L] → [1, N, C, L]
  // Here N=C=8 satisfies the constraint for the kernel=1 optimization
  //===--------------------------------------------------------------------===//
  func.func @test_conv1d_kernel1(%arg0: tensor<8x8x64x!quant.uniform<i8:f32, 0.1:0>>,
                                 %arg1: tensor<32x8x1x!quant.uniform<i8:f32, 0.05:0>>)
                                 -> tensor<8x32x64x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1],
      group = 1 : si64,
      kernel_shape = [1],
      pads = [0, 0],
      strides = [1]
    } : (tensor<8x8x64x!quant.uniform<i8:f32, 0.1:0>>,
         tensor<32x8x1x!quant.uniform<i8:f32, 0.05:0>>, none)
         -> tensor<8x32x64x!quant.uniform<i8:f32, 0.1:0>>
    return %1 : tensor<8x32x64x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_conv1d_kernel1
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x8x8x64x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.Reshape{{.*}} -> tensor<32x8x1x1x!quant.uniform<i8:f32, 5.000000e-02>>
  // CHECK: onnx.Conv{{.*}}kernel_shape = [1, 1]
  // CHECK: onnx.Reshape{{.*}} -> tensor<8x32x64x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 4: Conv1D with bias
  //===--------------------------------------------------------------------===//
  func.func @test_conv1d_bias(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
                              %arg1: tensor<32x16x3x!quant.uniform<i8:f32, 0.05:0>>,
                              %arg2: tensor<32xf32>)
                              -> tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>> {
    %1 = "onnx.Conv"(%arg0, %arg1, %arg2) {
      auto_pad = "NOTSET",
      dilations = [1],
      group = 1 : si64,
      kernel_shape = [3],
      pads = [0, 0],
      strides = [1]
    } : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
         tensor<32x16x3x!quant.uniform<i8:f32, 0.05:0>>, tensor<32xf32>)
         -> tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>>
    return %1 : tensor<1x32x62x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_conv1d_bias
  // CHECK: onnx.Reshape
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK: onnx.Reshape

  //===--------------------------------------------------------------------===//
  // Test 5: DepthwiseConv1D without activation
  // group == input_channels (depthwise)
  //===--------------------------------------------------------------------===//
  func.func @test_dwconv1d(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
                           %arg1: tensor<16x1x3x!quant.uniform<i8:f32, 0.05:0>>)
                           -> tensor<1x16x62x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1],
      group = 16 : si64,
      kernel_shape = [3],
      pads = [0, 0],
      strides = [1]
    } : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
         tensor<16x1x3x!quant.uniform<i8:f32, 0.05:0>>, none)
         -> tensor<1x16x62x!quant.uniform<i8:f32, 0.1:0>>
    return %1 : tensor<1x16x62x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_dwconv1d
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x1x64x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.Reshape{{.*}} -> tensor<16x1x1x3x!quant.uniform<i8:f32, 5.000000e-02>>
  // CHECK: onnx.Conv{{.*}}group = 16{{.*}}kernel_shape = [1, 3]
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x62x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 6: DepthwiseConv1D with Relu
  //===--------------------------------------------------------------------===//
  func.func @test_dwconv1d_relu(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
                                %arg1: tensor<16x1x3x!quant.uniform<i8:f32, 0.05:0>>)
                                -> tensor<1x16x62x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1],
      group = 16 : si64,
      kernel_shape = [3],
      pads = [0, 0],
      strides = [1]
    } : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
         tensor<16x1x3x!quant.uniform<i8:f32, 0.05:0>>, none)
         -> tensor<1x16x62x!quant.uniform<i8:f32, 0.1:0>>
    %2 = "onnx.Relu"(%1) : (tensor<1x16x62x!quant.uniform<i8:f32, 0.1:0>>)
                           -> tensor<1x16x62x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x16x62x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_dwconv1d_relu
  // CHECK: onnx.Reshape
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv{{.*}}group = 16
  // CHECK: onnx.Relu
  // CHECK: onnx.Reshape

  //===--------------------------------------------------------------------===//
  // Test 7: MaxPool1D without activation
  // Input: [N, C, L] = [1, 16, 64] → [N, C, 1, L] = [1, 16, 1, 64]
  //===--------------------------------------------------------------------===//
  func.func @test_maxpool1d(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>)
                            -> tensor<1x16x32x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [2],
      pads = [0, 0],
      strides = [2]
    } : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>)
        -> tensor<1x16x32x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<1x16x32x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_maxpool1d
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x1x64x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.MaxPoolSingleOut{{.*}}kernel_shape = [1, 2]{{.*}}pads = [0, 0, 0, 0]{{.*}}strides = [1, 2]
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x32x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 8: Conv1D with padding and stride
  //===--------------------------------------------------------------------===//
  func.func @test_conv1d_pad_stride(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
                                    %arg1: tensor<32x16x3x!quant.uniform<i8:f32, 0.05:0>>)
                                    -> tensor<1x32x32x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1],
      group = 1 : si64,
      kernel_shape = [3],
      pads = [1, 1],
      strides = [2]
    } : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>,
         tensor<32x16x3x!quant.uniform<i8:f32, 0.05:0>>, none)
         -> tensor<1x32x32x!quant.uniform<i8:f32, 0.1:0>>
    return %1 : tensor<1x32x32x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_conv1d_pad_stride
  // CHECK: onnx.Reshape
  // CHECK: onnx.Reshape
  // CHECK: onnx.Conv{{.*}}pads = [0, 1, 0, 1]{{.*}}strides = [1, 2]
  // CHECK: onnx.Reshape

  //===--------------------------------------------------------------------===//
  // Test 9: Should NOT match - 2D convolution (4D input)
  //===--------------------------------------------------------------------===//
  func.func @test_conv2d_no_match(%arg0: tensor<1x16x64x64xf32>,
                                  %arg1: tensor<32x16x3x3xf32>) -> tensor<1x32x62x62xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
    } : (tensor<1x16x64x64xf32>, tensor<32x16x3x3xf32>, none) -> tensor<1x32x62x62xf32>
    return %1 : tensor<1x32x62x62xf32>
  }
  // CHECK-LABEL: func.func @test_conv2d_no_match
  // CHECK-NOT: onnx.Reshape
  // CHECK: onnx.Conv
  // CHECK-NOT: onnx.Reshape

  //===--------------------------------------------------------------------===//
  // Test 10: Should NOT match - 2D MaxPool (4D input)
  //===--------------------------------------------------------------------===//
  func.func @test_maxpool2d_no_match(%arg0: tensor<1x16x64x64xf32>) -> tensor<1x16x32x32xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [2, 2],
      pads = [0, 0, 0, 0],
      strides = [2, 2]
    } : (tensor<1x16x64x64xf32>) -> tensor<1x16x32x32xf32>
    return %0 : tensor<1x16x32x32xf32>
  }
  // CHECK-LABEL: func.func @test_maxpool2d_no_match
  // CHECK-NOT: onnx.Reshape
  // CHECK: onnx.MaxPoolSingleOut
  // CHECK-NOT: onnx.Reshape
}
