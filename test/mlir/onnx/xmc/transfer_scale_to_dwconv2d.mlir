// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt %s -transfer-scale-to-dwconv2d -o - | FileCheck %s

module {
  //===--------------------------------------------------------------------===//
  // Test 1: Scale with 1D weight (basic case)
  //===--------------------------------------------------------------------===//
  func.func @test_scale_1d(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>)
                           -> tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>> {
    %scale = "onnx.Constant"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
    %0 = "onnx.Mul"(%arg0, %scale) : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>, tensor<64xf32>) 
         -> tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_scale_1d
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x1x16x64x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.Reshape{{.*}} -> tensor<64x1x1x1xf32>
  // CHECK: onnx.Conv{{.*}}group = 1{{.*}}kernel_shape = [1, 1]{{.*}}pads = [0, 0, 0, 0]{{.*}}strides = [1, 1]
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x64x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 2: Scale with activation (relu)
  //===--------------------------------------------------------------------===//
  func.func @test_scale_with_relu(%arg0: tensor<1x32x128x!quant.uniform<i8:f32, 0.05:0>>)
                                  -> tensor<1x32x128x!quant.uniform<i8:f32, 0.05:0>> {
    %scale = "onnx.Constant"() {value = dense<0.5> : tensor<128xf32>} : () -> tensor<128xf32>
    %0 = "onnx.Mul"(%arg0, %scale) : (tensor<1x32x128x!quant.uniform<i8:f32, 0.05:0>>, tensor<128xf32>) 
         -> tensor<1x32x128x!quant.uniform<i8:f32, 0.05:0>>
    %1 = "onnx.Relu"(%0) : (tensor<1x32x128x!quant.uniform<i8:f32, 0.05:0>>) 
         -> tensor<1x32x128x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x32x128x!quant.uniform<i8:f32, 0.05:0>>
  }
  // CHECK-LABEL: func.func @test_scale_with_relu
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x1x32x128x!quant.uniform<i8:f32, 5.000000e-02>>
  // CHECK: onnx.Reshape{{.*}} -> tensor<128x1x1x1xf32>
  // CHECK: onnx.Conv{{.*}}group = 1{{.*}}kernel_shape = [1, 1]
  // CHECK: onnx.Relu
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x32x128x!quant.uniform<i8:f32, 5.000000e-02>>

  //===--------------------------------------------------------------------===//
  // Test 3: Scale with 2D input
  //===--------------------------------------------------------------------===//
  func.func @test_scale_2d(%arg0: tensor<8x256x!quant.uniform<i8:f32, 0.1:0>>)
                           -> tensor<8x256x!quant.uniform<i8:f32, 0.1:0>> {
    %scale = "onnx.Constant"() {value = dense<1.5> : tensor<256xf32>} : () -> tensor<256xf32>
    %0 = "onnx.Mul"(%arg0, %scale) : (tensor<8x256x!quant.uniform<i8:f32, 0.1:0>>, tensor<256xf32>) 
         -> tensor<8x256x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<8x256x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_scale_2d
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x1x8x256x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: onnx.Reshape{{.*}} -> tensor<256x1x1x1xf32>
  // CHECK: onnx.Conv{{.*}}group = 1{{.*}}kernel_shape = [1, 1]{{.*}}pads = [0, 0, 0, 0]{{.*}}strides = [1, 1]
  // CHECK: onnx.Reshape{{.*}} -> tensor<8x256x!quant.uniform<i8:f32, 1.000000e-01>>

  //===--------------------------------------------------------------------===//
  // Test 4: Scale with 4D input (no reshape needed)
  //===--------------------------------------------------------------------===//
  func.func @test_scale_4d(%arg0: tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>>)
                           -> tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>> {
    %scale = "onnx.Constant"() {value = dense<2.0> : tensor<64xf32>} : () -> tensor<64xf32>
    %0 = "onnx.Mul"(%arg0, %scale) : (tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>>, tensor<64xf32>) 
         -> tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_scale_4d
  // CHECK: onnx.Reshape{{.*}} -> tensor<64x1x1x1xf32>
  // CHECK: onnx.Conv{{.*}}group = 16{{.*}}kernel_shape = [1, 1]{{.*}}pads = [0, 0, 0, 0]{{.*}}strides = [1, 1]
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x32x64x!quant.uniform<i8:f32, 1.000000e-01>>
}
