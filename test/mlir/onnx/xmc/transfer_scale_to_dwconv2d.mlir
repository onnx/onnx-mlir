// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt %s -transfer-scale-to-dwconv2d -o - | FileCheck %s

module {
  //===--------------------------------------------------------------------===//
  // Test 1: Scale with quantized input/output — should NOT be converted
  // (matches golden behavior: qdq_enabled skips the pass)
  //===--------------------------------------------------------------------===//
  func.func @test_scale_1d_quant_skip(%arg0: tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>)
                           -> tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>> {
    %scale = "onnx.Constant"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
    %0 = "onnx.Mul"(%arg0, %scale) : (tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>, tensor<64xf32>) 
         -> tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<1x16x64x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_scale_1d_quant_skip
  // CHECK: onnx.Mul
  // CHECK-NOT: onnx.XFEConv

  //===--------------------------------------------------------------------===//
  // Test 2: Scale with float input/output — should be converted
  //===--------------------------------------------------------------------===//
  func.func @test_scale_1d_float(%arg0: tensor<1x16x64xf32>)
                           -> tensor<1x16x64xf32> {
    %scale = "onnx.Constant"() {value = dense<1.0> : tensor<64xf32>} : () -> tensor<64xf32>
    %0 = "onnx.Mul"(%arg0, %scale) : (tensor<1x16x64xf32>, tensor<64xf32>) 
         -> tensor<1x16x64xf32>
    return %0 : tensor<1x16x64xf32>
  }
  // CHECK-LABEL: func.func @test_scale_1d_float
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x1x16x64xf32>
  // CHECK: onnx.Reshape{{.*}} -> tensor<64x1x1x1xf32>
  // CHECK: onnx.XFEConv{{.*}}group = 64{{.*}}kernel_shape = [1, 1]{{.*}}pads = [0, 0, 0, 0]{{.*}}strides = [1, 1]
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x64xf32>

  //===--------------------------------------------------------------------===//
  // Test 3: Scale with float 4D input — should be converted
  //===--------------------------------------------------------------------===//
  func.func @test_scale_4d_float(%arg0: tensor<1x16x32x64xf32>)
                           -> tensor<1x16x32x64xf32> {
    %scale = "onnx.Constant"() {value = dense<2.0> : tensor<64xf32>} : () -> tensor<64xf32>
    %0 = "onnx.Mul"(%arg0, %scale) : (tensor<1x16x32x64xf32>, tensor<64xf32>) 
         -> tensor<1x16x32x64xf32>
    return %0 : tensor<1x16x32x64xf32>
  }
  // CHECK-LABEL: func.func @test_scale_4d_float
  // CHECK: onnx.Reshape{{.*}} -> tensor<64x1x1x1xf32>
  // CHECK: onnx.XFEConv{{.*}}group = 64{{.*}}kernel_shape = [1, 1]{{.*}}pads = [0, 0, 0, 0]{{.*}}strides = [1, 1]
  // CHECK: onnx.Reshape{{.*}} -> tensor<1x16x32x64xf32>

  //===--------------------------------------------------------------------===//
  // Test 4: Scale with quantized 4D input — should NOT be converted
  //===--------------------------------------------------------------------===//
  func.func @test_scale_4d_quant_skip(%arg0: tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>>)
                           -> tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>> {
    %scale = "onnx.Constant"() {value = dense<2.0> : tensor<64xf32>} : () -> tensor<64xf32>
    %0 = "onnx.Mul"(%arg0, %scale) : (tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>>, tensor<64xf32>) 
         -> tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>>
    return %0 : tensor<1x16x32x64x!quant.uniform<i8:f32, 0.1:0>>
  }
  // CHECK-LABEL: func.func @test_scale_4d_quant_skip
  // CHECK: onnx.Mul
  // CHECK-NOT: onnx.XFEConv
}
