// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --transform-reshapelike-op-to-reshape %s | FileCheck %s

module {
  // Test 1: Flatten
  func.func @flatten_4d_axis_1(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x60xf32> {
    %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<2x3x4x5xf32>) -> tensor<2x60xf32>
    return %0 : tensor<2x60xf32>
  }
  // CHECK-LABEL: func.func @flatten_4d_axis_1
  // CHECK-NOT: onnx.Flatten
  // CHECK: onnx.Constant
  // CHECK: onnx.Reshape
  // CHECK-SAME: tensor<2x60xf32>

  // Test 2: Trivial Transpose (should convert to reshape)
  func.func @trivial_transpose(%arg0: tensor<1x3x1x4xf32>) -> tensor<1x1x3x4xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x3x1x4xf32>) -> tensor<1x1x3x4xf32>
    return %0 : tensor<1x1x3x4xf32>
  }
  // CHECK-LABEL: func.func @trivial_transpose
  // CHECK-NOT: onnx.Transpose
  // CHECK: onnx.Constant
  // CHECK: onnx.Reshape
  // CHECK-SAME: tensor<1x1x3x4xf32>

  // Test 3: Non-trivial Transpose (should NOT convert)
  func.func @non_trivial_transpose(%arg0: tensor<2x3x4xf32>) -> tensor<2x4x3xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<2x3x4xf32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
  // CHECK-LABEL: func.func @non_trivial_transpose
  // CHECK: onnx.Transpose
  // CHECK-NOT: onnx.Reshape

  // Test 4: Squeeze with specific axes
  func.func @squeeze_axes(%arg0: tensor<1x3x1x4xf32>) -> tensor<1x3x4xf32> {
    %axes = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Squeeze"(%arg0, %axes) : (tensor<1x3x1x4xf32>, tensor<1xi64>) -> tensor<1x3x4xf32>
    return %0 : tensor<1x3x4xf32>
  }
  // CHECK-LABEL: func.func @squeeze_axes
  // CHECK-NOT: onnx.Squeeze
  // CHECK: onnx.Constant
  // CHECK: onnx.Reshape
  // CHECK-SAME: tensor<1x3x4xf32>

  // Test 5: Unsqueeze with specific axes
  func.func @unsqueeze_axes(%arg0: tensor<3x4xf32>) -> tensor<1x3x1x4xf32> {
    %axes = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %0 = "onnx.Unsqueeze"(%arg0, %axes) : (tensor<3x4xf32>, tensor<2xi64>) -> tensor<1x3x1x4xf32>
    return %0 : tensor<1x3x1x4xf32>
  }
  // CHECK-LABEL: func.func @unsqueeze_axes
  // CHECK-NOT: onnx.Unsqueeze
  // CHECK: onnx.Constant
  // CHECK: onnx.Reshape
  // CHECK-SAME: tensor<1x3x1x4xf32>

  // Test 6: Flatten with quantized type
  func.func @flatten_4d_axis_1_quant(%arg0: tensor<2x3x4x5x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<2x60x!quant.uniform<u8:f32, 0.20000000298023224:1>> {
    %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<2x3x4x5x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<2x60x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    return %0 : tensor<2x60x!quant.uniform<u8:f32, 0.20000000298023224:1>>
  }
  // CHECK-LABEL: func.func @flatten_4d_axis_1_quant
  // CHECK-NOT: onnx.Flatten
  // CHECK: onnx.Constant
  // CHECK: onnx.Reshape

  // Test 7: Trivial Transpose with quantized type
  func.func @trivial_transpose_quant(%arg0: tensor<1x3x1x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<1x1x3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x3x1x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<1x1x3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    return %0 : tensor<1x1x3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
  }
  // CHECK-LABEL: func.func @trivial_transpose_quant
  // CHECK-NOT: onnx.Transpose
  // CHECK: onnx.Constant
  // CHECK: onnx.Reshape

  // Test 8: Squeeze with quantized type
  func.func @squeeze_axes_quant(%arg0: tensor<1x3x1x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<1x3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>> {
    %axes = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Squeeze"(%arg0, %axes) : (tensor<1x3x1x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>, tensor<1xi64>) -> tensor<1x3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    return %0 : tensor<1x3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
  }
  // CHECK-LABEL: func.func @squeeze_axes_quant
  // CHECK-NOT: onnx.Squeeze
  // CHECK: onnx.Constant
  // CHECK: onnx.Reshape

  // Test 9: Unsqueeze with quantized type
  func.func @unsqueeze_axes_quant(%arg0: tensor<3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>) -> tensor<1x3x1x4x!quant.uniform<u8:f32, 0.20000000298023224:1>> {
    %axes = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %0 = "onnx.Unsqueeze"(%arg0, %axes) : (tensor<3x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>, tensor<2xi64>) -> tensor<1x3x1x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
    return %0 : tensor<1x3x1x4x!quant.uniform<u8:f32, 0.20000000298023224:1>>
  }
  // CHECK-LABEL: func.func @unsqueeze_axes_quant
  // CHECK-NOT: onnx.Unsqueeze
  // CHECK: onnx.Constant
  // CHECK: onnx.Reshape
}

