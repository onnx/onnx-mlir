// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt %s -transfer-op1d-to-op2d -o - | FileCheck %s

module {
  //===--------------------------------------------------------------------===//
  // Test 1: Conv1D
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
  // Test 2: ConvTranspose1D
  // Output calculation: (32-1)*2 + 3 = 65
  //===--------------------------------------------------------------------===//
  func.func @test_convtranspose1d(%arg0: tensor<1x16x32x!quant.uniform<i8:f32, 0.1:0>>,
                                  %arg1: tensor<16x32x3x!quant.uniform<i8:f32, 0.05:0>>)
                                  -> tensor<1x32x65x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {
      auto_pad = "NOTSET",
      dilations = [1],
      group = 1 : si64,
      kernel_shape = [3],
      pads = [0, 0],
      strides = [2]
    } : (tensor<1x16x32x!quant.uniform<i8:f32, 0.1:0>>,
         tensor<16x32x3x!quant.uniform<i8:f32, 0.05:0>>, none)
         -> tensor<1x32x65x!quant.uniform<i8:f32, 0.1:0>>
    return %1 : tensor<1x32x65x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_convtranspose1d
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x1x32x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.Reshape{{.*}} -> tensor<16x32x1x3x!quant.uniform<i8:f32, 5.000000e-02>>
  // CHECK: onnx.ConvTranspose{{.*}}kernel_shape = [1, 3]{{.*}}pads = [0, 0, 0, 0]{{.*}}strides = [1, 2]
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x32x65x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 3: MaxPool1D
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
  // Test 4: AveragePool1D
  //===--------------------------------------------------------------------===//
  func.func @test_avgpool1d(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>)
                            -> tensor<1x16x32x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      kernel_shape = [2],
      pads = [0, 0],
      strides = [2]
    } : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>)
        -> tensor<1x16x32x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<1x16x32x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_avgpool1d
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x1x64x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.AveragePool{{.*}}kernel_shape = [1, 2]{{.*}}pads = [0, 0, 0, 0]{{.*}}strides = [1, 2]
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x32x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 5: GlobalMaxPool1D
  //===--------------------------------------------------------------------===//
  func.func @test_globalmaxpool1d(%arg0: tensor<2x64x256x!quant.uniform<i8:f32, 0.1:0>>)
                                  -> tensor<2x64x1x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<2x64x256x!quant.uniform<i8:f32, 0.1:0>>)
                                       -> tensor<2x64x1x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<2x64x1x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_globalmaxpool1d
  // CHECK: onnx.Reshape{{.*}} -> tensor<2x64x1x256x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.GlobalMaxPool
  // CHECK: onnx.Reshape{{.*}} -> tensor<2x64x1x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 6: GlobalAveragePool1D
  //===--------------------------------------------------------------------===//
  func.func @test_globalavgpool1d(%arg0: tensor<4x128x512x!quant.uniform<i8:f32, 0.05:0>>)
                                  -> tensor<4x128x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<4x128x512x!quant.uniform<i8:f32, 0.05:0>>)
                                           -> tensor<4x128x1x!quant.uniform<i8:f32, 0.05:0>>
    return %0 : tensor<4x128x1x!quant.uniform<i8:f32, 0.05:0>>
  }
  // CHECK-LABEL: func.func @test_globalavgpool1d
  // CHECK: onnx.Reshape{{.*}} -> tensor<4x128x1x512x!quant.uniform<i8:f32, 5.000000e-02>>
  // CHECK: onnx.GlobalAveragePool
  // CHECK: onnx.Reshape{{.*}} -> tensor<4x128x1x!quant.uniform<i8:f32, 5.000000e-02>>
}
