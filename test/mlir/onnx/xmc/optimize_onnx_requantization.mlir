// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file %s --optimize-onnx-requantization | FileCheck %s

// Test 1: Reshape with requantization - parent operation should be updated
func.func @test_reshape_requantization(%arg0: tensor<1x8x64xui8>, %arg1: tensor<1x8x64xui8>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>> {
  %scale1 = onnx.Constant dense<5.000000e-02> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>

  %0 = "onnx.DequantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %1 = "onnx.DequantizeLinear"(%arg1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  // Add produces quantized output with scale=0.1, zp=0
  %2 = "onnx.Add"(%0, %1) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // Reshape with different quantization params: scale=0.2, zp=5
  %3 = "onnx.Reshape"(%2, %shape) {allowzero = 0 : si64} : (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>

  return %3 : tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>
}

// CHECK-LABEL: func.func @test_reshape_requantization
// CHECK: %[[ADD:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ADD]]
// CHECK-SAME: (tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>

// -----

// Test 2: Transpose with requantization
func.func @test_transpose_requantization(%arg0: tensor<1x8x64xui8>, %arg1: tensor<1x8x64xui8>) -> tensor<1x8x4x16x!quant.uniform<u8:f32, 0.20000000298023224:5>> {
  %scale1 = onnx.Constant dense<5.000000e-02> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>

  %0 = "onnx.DequantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %1 = "onnx.DequantizeLinear"(%arg1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  %2 = "onnx.Add"(%0, %1) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64xf32>
  %3 = "onnx.Reshape"(%2, %shape) {allowzero = 0 : si64} : (tensor<1x8x64xf32>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.10000000149011612>>

  // Transpose with different quantization params: scale=0.2, zp=5
  %4 = "onnx.Transpose"(%3) {perm = [0, 1, 3, 2]} : (tensor<1x8x16x4x!quant.uniform<u8:f32, 0.10000000149011612>>) -> tensor<1x8x4x16x!quant.uniform<u8:f32, 0.20000000298023224:5>>

  return %4 : tensor<1x8x4x16x!quant.uniform<u8:f32, 0.20000000298023224:5>>
}

// CHECK-LABEL: func.func @test_transpose_requantization
// CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"
// CHECK-SAME: -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[RESHAPE]])
// CHECK-SAME: (tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>) -> tensor<1x8x4x16x!quant.uniform<u8:f32, 0.20000000298023224:5>>

// -----

// Test 3: Chain propagation - Add → Reshape → Transpose
func.func @test_chain_propagation(%arg0: tensor<1x8x64xui8>, %arg1: tensor<1x8x64xui8>) -> tensor<1x8x4x16x!quant.uniform<u8:f32, 0.20000000298023224:5>> {
  %scale1 = onnx.Constant dense<5.000000e-02> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>

  %0 = "onnx.DequantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %1 = "onnx.DequantizeLinear"(%arg1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  // Add with scale=0.1, zp=0
  %2 = "onnx.Add"(%0, %1) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // Reshape with scale=0.15, zp=3
  %3 = "onnx.Reshape"(%2, %shape) {allowzero = 0 : si64} : (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.15000000298023224:3>>

  // Transpose with scale=0.2, zp=5
  %4 = "onnx.Transpose"(%3) {perm = [0, 1, 3, 2]} : (tensor<1x8x16x4x!quant.uniform<u8:f32, 0.15000000298023224:3>>) -> tensor<1x8x4x16x!quant.uniform<u8:f32, 0.20000000298023224:5>>

  return %4 : tensor<1x8x4x16x!quant.uniform<u8:f32, 0.20000000298023224:5>>
}

// CHECK-LABEL: func.func @test_chain_propagation
// CHECK: %[[ADD:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ADD]]
// CHECK-SAME: (tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[RESHAPE]])
// CHECK-SAME: (tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>) -> tensor<1x8x4x16x!quant.uniform<u8:f32, 0.20000000298023224:5>>

// -----

// Test 4: No requantization - operations with same quantization parameters
func.func @test_no_requantization(%arg0: tensor<1x8x64xui8>, %arg1: tensor<1x8x64xui8>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.10000000149011612>> {
  %scale1 = onnx.Constant dense<5.000000e-02> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>

  %0 = "onnx.DequantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %1 = "onnx.DequantizeLinear"(%arg1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  // Add and Reshape with same quantization params
  %2 = "onnx.Add"(%0, %1) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
  %3 = "onnx.Reshape"(%2, %shape) {allowzero = 0 : si64} : (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.10000000149011612>>

  return %3 : tensor<1x8x16x4x!quant.uniform<u8:f32, 0.10000000149011612>>
}

// CHECK-LABEL: func.func @test_no_requantization
// CHECK: %[[ADD:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
// CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ADD]]
// CHECK-SAME: (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.10000000149011612>>

// -----

// Test 5: Multiple uses - parent operation with multiple consumers
func.func @test_multiple_uses(%arg0: tensor<1x8x64xui8>, %arg1: tensor<1x8x64xui8>) -> (tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>, tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>) {
  %scale1 = onnx.Constant dense<5.000000e-02> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>

  %0 = "onnx.DequantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %1 = "onnx.DequantizeLinear"(%arg1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  // Add has multiple uses - should NOT be updated
  %2 = "onnx.Add"(%0, %1) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
  %3 = "onnx.Reshape"(%2, %shape) {allowzero = 0 : si64} : (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>

  return %3, %2 : tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>, tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
}

// CHECK-LABEL: func.func @test_multiple_uses
// CHECK: %[[ADD:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
// CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ADD]]
// CHECK-SAME: (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<4xi64>) -> tensor<1x8x16x4x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: return %[[RESHAPE]], %[[ADD]]

// -----

// Test 6: Concat with requantization - two inputs with different quantization
func.func @test_concat_requantization(%arg0: tensor<1x8x64xui8>, %arg1: tensor<1x8x64xui8>) -> tensor<1x16x64x!quant.uniform<u8:f32, 0.20000000298023224:5>> {
  %scale1 = onnx.Constant dense<5.000000e-02> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>

  %0 = "onnx.DequantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %1 = "onnx.DequantizeLinear"(%arg1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  // Two Add operations producing quantized outputs with scale=0.1, zp=0
  %2 = "onnx.Add"(%0, %0) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
  %3 = "onnx.Add"(%1, %1) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // Concat with different quantization params: scale=0.2, zp=5
  %4 = "onnx.Concat"(%2, %3) {axis = 1 : si64} : (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>) -> tensor<1x16x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>

  return %4 : tensor<1x16x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
}

// CHECK-LABEL: func.func @test_concat_requantization
// CHECK: %[[ADD1:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[ADD2:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%[[ADD1]], %[[ADD2]])
// CHECK-SAME: (tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>, tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>) -> tensor<1x16x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>

// -----

// Test 7: Concat with three inputs all having single use
func.func @test_concat_three_inputs(%arg0: tensor<1x8x64xf32>) -> tensor<1x24x64x!quant.uniform<u8:f32, 0.20000000298023224:5>> {
  // Three operations producing quantized outputs with scale=0.1, zp=0
  %0 = "onnx.Relu"(%arg0) : (tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
  %1 = "onnx.Sigmoid"(%arg0) : (tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
  %2 = "onnx.Tanh"(%arg0) : (tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // Concat with different quantization: scale=0.2, zp=5
  %3 = "onnx.Concat"(%0, %1, %2) {axis = 1 : si64} : (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>) -> tensor<1x24x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>

  return %3 : tensor<1x24x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
}

// CHECK-LABEL: func.func @test_concat_three_inputs
// CHECK: %[[RELU:.*]] = "onnx.Relu"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[SIGMOID:.*]] = "onnx.Sigmoid"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[TANH:.*]] = "onnx.Tanh"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%[[RELU]], %[[SIGMOID]], %[[TANH]])

// -----

// Test 8: Concat where one parent has multiple uses (partial optimization)
func.func @test_concat_multiple_uses(%arg0: tensor<1x8x64xf32>) -> (tensor<1x16x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>, tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>) {
  // Two operations producing quantized outputs with scale=0.1, zp=0
  %0 = "onnx.Relu"(%arg0) : (tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
  %1 = "onnx.Sigmoid"(%arg0) : (tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // Concat with different quantization: scale=0.2, zp=5
  %2 = "onnx.Concat"(%0, %1) {axis = 1 : si64} : (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>) -> tensor<1x16x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>

  // %1 has multiple uses - should NOT be updated
  return %2, %1 : tensor<1x16x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>, tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
}

// CHECK-LABEL: func.func @test_concat_multiple_uses
// CHECK: %[[RELU:.*]] = "onnx.Relu"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[SIGMOID:.*]] = "onnx.Sigmoid"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
// CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%[[RELU]], %[[SIGMOID]])
// CHECK: return %[[CONCAT]], %[[SIGMOID]]

// -----

// Test 9: Slice with requantization - parent operation should be updated
func.func @test_slice_requantization(%arg0: tensor<1x8x64xui8>, %arg1: tensor<1x8x64xui8>) -> tensor<1x4x64x!quant.uniform<u8:f32, 0.20000000298023224:5>> {
  %scale1 = onnx.Constant dense<5.000000e-02> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %starts = onnx.Constant dense<[0, 0, 0]> : tensor<3xi64>
  %ends = onnx.Constant dense<[1, 4, 64]> : tensor<3xi64>
  %axes = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
  %steps = onnx.Constant dense<[1, 1, 1]> : tensor<3xi64>

  %0 = "onnx.DequantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %1 = "onnx.DequantizeLinear"(%arg1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  // Add produces quantized output with scale=0.1, zp=0
  %2 = "onnx.Add"(%0, %1) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  // Slice with different quantization params: scale=0.2, zp=5
  %3 = "onnx.Slice"(%2, %starts, %ends, %axes, %steps) : (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>

  return %3 : tensor<1x4x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
}

// CHECK-LABEL: func.func @test_slice_requantization
// CHECK: %[[ADD:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>
// CHECK: %[[SLICE:.*]] = "onnx.Slice"(%[[ADD]]
// CHECK-SAME: (tensor<1x8x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x64x!quant.uniform<u8:f32, 0.20000000298023224:5>>

// -----

// Test 10: Slice with no requantization
func.func @test_slice_no_requantization(%arg0: tensor<1x8x64xui8>, %arg1: tensor<1x8x64xui8>) -> tensor<1x4x64x!quant.uniform<u8:f32, 0.10000000149011612>> {
  %scale1 = onnx.Constant dense<5.000000e-02> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %starts = onnx.Constant dense<[0, 0, 0]> : tensor<3xi64>
  %ends = onnx.Constant dense<[1, 4, 64]> : tensor<3xi64>
  %axes = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
  %steps = onnx.Constant dense<[1, 1, 1]> : tensor<3xi64>

  %0 = "onnx.DequantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %1 = "onnx.DequantizeLinear"(%arg1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  // Add and Slice with same quantization params
  %2 = "onnx.Add"(%0, %1) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
  %3 = "onnx.Slice"(%2, %starts, %ends, %axes, %steps) : (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x64x!quant.uniform<u8:f32, 0.10000000149011612>>

  return %3 : tensor<1x4x64x!quant.uniform<u8:f32, 0.10000000149011612>>
}

// CHECK-LABEL: func.func @test_slice_no_requantization
// CHECK: %[[ADD:.*]] = "onnx.Add"
// CHECK-SAME: -> tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>
// CHECK: %[[SLICE:.*]] = "onnx.Slice"(%[[ADD]]
// CHECK-SAME: (tensor<1x8x64x!quant.uniform<u8:f32, 0.10000000149011612>>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x64x!quant.uniform<u8:f32, 0.10000000149011612>>
