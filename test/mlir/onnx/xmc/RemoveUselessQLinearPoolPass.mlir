// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --remove_useless_qlinear_pool %s | FileCheck %s

// Test MaxPool fulfilling noop criteria is removed
// CHECK-LABEL: func.func @noop_linear_maxpool
func.func @noop_linear_maxpool(%arg0: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>) {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {
      kernel_shape = [1, 1],
      strides = [1, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %0 : tensor<1x3x4x4xf32>

  }

// CHECK: return %arg0 : tensor<1x3x4x4xf32>
// CHECK-NOT: onnx.MaxPoolSingleOut

// Test AveragePool fulfilling noop criteria is removed
// CHECK-LABEL: func.func @noop_linear_avgpool
func.func @noop_linear_avgpool(%arg0: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>) {
     %0 = "onnx.AveragePool"(%arg0) {
      kernel_shape = [1, 1],
      strides = [1, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %0 : tensor<1x3x4x4xf32>
  }

// CHECK: return %arg0 : tensor<1x3x4x4xf32>
// CHECK-NOT: onnx.AveragePool


// Test AveragePool with strides != 1,1 is not removed
// CHECK-LABEL: func.func @linear_avgpool
func.func @linear_avgpool(%arg0: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>) {
     %0 = "onnx.AveragePool"(%arg0) {
      kernel_shape = [1, 1],
      strides = [2, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    return %0 : tensor<1x3x4x4xf32>
  }

// CHECK: %[[POOL:.*]] = "onnx.AveragePool"(%arg0)
// CHECK-SAME: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
// CHECK: return %[[POOL]] : tensor<1x3x4x4xf32>

// Test quantized MaxPool fulfilling noop criteria is removed
// CHECK-LABEL: func.func @noop_quantized_linear_pool
func.func @noop_quantized_linear_pool(%arg0: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>) {
    %0 = onnx.Constant dense<2.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<1> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) : (tensor<1x3x4x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x3x4x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    %3 = "onnx.MaxPoolSingleOut"(%2) {
      kernel_shape = [1, 1],
      strides = [1, 1]} : (tensor<1x3x4x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<1x3x4x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    %4 = "onnx.DequantizeLinear"(%3, %0, %1) : (tensor<1x3x4x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>, tensor<f32>, tensor<i8>) -> tensor<1x3x4x4xf32>
    return %4 : tensor<1x3x4x4xf32>
  }

// CHECK-NOT: "onnx.MaxPoolSingleOut"
// CHECK: "onnx.DequantizeLinear"(%2


// Test quantized MaxPool with different scales in in/out tensors is not removed
// CHECK-LABEL: func.func @qscale_qlinear_pool
func.func @qscale_qlinear_pool(%arg0: tensor<1x3x4x4xf32>) -> (tensor<1x3x4x4xf32>) {
    %0 = onnx.Constant dense<2.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<1> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) : (tensor<1x3x4x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x3x4x4x!quant.uniform<u8:f32, 0.10000000298023224:1>>
    %3 = "onnx.MaxPoolSingleOut"(%2) {
      kernel_shape = [1, 1],
      strides = [1, 1]} : (tensor<1x3x4x4x!quant.uniform<u8:f32, 0.10000000298023224:1>>) -> tensor<1x3x4x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    %4 = "onnx.DequantizeLinear"(%3, %0, %1) : (tensor<1x3x4x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>, tensor<f32>, tensor<i8>) -> tensor<1x3x4x4xf32>
    return %4 : tensor<1x3x4x4xf32>
  }


// CHECK: %[[POOL:.*]] = "onnx.MaxPoolSingleOut"(%2)
// CHECK-SAME: tensor<1x3x4x4x!quant.uniform<u8:f32, 0.10000000298023223:1>>) -> tensor<1x3x4x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[POOL]]
// CHECK: return %[[DQ]] : tensor<1x3x4x4xf32>
