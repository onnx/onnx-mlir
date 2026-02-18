// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file %s --optimize-onnx-requantization | FileCheck %s

// ==========================================================================
// OnnxQDQRequantizationOptimizationPattern Tests
// Pattern: Q(s1,zp1) -> DQ(s1,zp1) -> DataMovementOp -> Q(s2,zp2)
// Result:  Q(s1,zp1) -> DQ(s2,zp2) -> DataMovementOp -> Q(s2,zp2)
// Only the DQ node is modified to use Q(output)'s scale/zp.
// ==========================================================================

// Test 1: Reshape with requantization - DQ should be updated to use Q(output)'s params
func.func @test_reshape_qdq_requantization(%arg0: tensor<1x8x64xf32>) -> tensor<1x8x16x4xui8> {
  %scale1 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %scale2 = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %zp2 = onnx.Constant dense<5> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>

  %0 = "onnx.QuantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xui8>
  %1 = "onnx.DequantizeLinear"(%0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %2 = "onnx.Reshape"(%1, %shape) {allowzero = 0 : si64} : (tensor<1x8x64xf32>, tensor<4xi64>) -> tensor<1x8x16x4xf32>
  %3 = "onnx.QuantizeLinear"(%2, %scale2, %zp2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x16x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x16x4xui8>

  return %3 : tensor<1x8x16x4xui8>
}

// CHECK-LABEL: func.func @test_reshape_qdq_requantization
// CHECK-DAG: %[[S1:.*]] = onnx.Constant dense<1.000000e-01>
// CHECK-DAG: %[[ZP1:.*]] = onnx.Constant dense<0>
// CHECK-DAG: %[[S2:.*]] = onnx.Constant dense<2.000000e-01>
// CHECK-DAG: %[[ZP2:.*]] = onnx.Constant dense<5>
// Q(parent) unchanged: uses s1/zp1
// CHECK: %[[Q_PARENT:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S1]], %[[ZP1]])
// CHECK-SAME: -> tensor<1x8x64xui8>
// DQ updated: uses s2/zp2 (was s1/zp1)
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q_PARENT]], %[[S2]], %[[ZP2]])
// CHECK-SAME: -> tensor<1x8x64xf32>
// CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[DQ]]
// CHECK-SAME: -> tensor<1x8x16x4xf32>
// Q(output) uses s2/zp2
// CHECK: %[[Q_OUT:.*]] = "onnx.QuantizeLinear"(%[[RESHAPE]], %[[S2]], %[[ZP2]])
// CHECK-SAME: -> tensor<1x8x16x4xui8>

// -----

// Test 2: Transpose with requantization
func.func @test_transpose_qdq_requantization(%arg0: tensor<1x8x16x4xf32>) -> tensor<1x8x4x16xui8> {
  %scale1 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %scale2 = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %zp2 = onnx.Constant dense<5> : tensor<ui8>

  %0 = "onnx.QuantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x16x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x16x4xui8>
  %1 = "onnx.DequantizeLinear"(%0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x16x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x16x4xf32>
  %2 = "onnx.Transpose"(%1) {perm = [0, 1, 3, 2]} : (tensor<1x8x16x4xf32>) -> tensor<1x8x4x16xf32>
  %3 = "onnx.QuantizeLinear"(%2, %scale2, %zp2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x4x16xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x4x16xui8>

  return %3 : tensor<1x8x4x16xui8>
}

// CHECK-LABEL: func.func @test_transpose_qdq_requantization
// CHECK-DAG: %[[S1:.*]] = onnx.Constant dense<1.000000e-01>
// CHECK-DAG: %[[S2:.*]] = onnx.Constant dense<2.000000e-01>
// CHECK-DAG: %[[ZP2:.*]] = onnx.Constant dense<5>
// CHECK: %[[Q_PARENT:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S1]]
// CHECK-SAME: -> tensor<1x8x16x4xui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q_PARENT]], %[[S2]], %[[ZP2]])
// CHECK-SAME: -> tensor<1x8x16x4xf32>
// CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[DQ]])
// CHECK-SAME: -> tensor<1x8x4x16xf32>
// CHECK: %[[Q_OUT:.*]] = "onnx.QuantizeLinear"(%[[TRANSPOSE]], %[[S2]], %[[ZP2]])
// CHECK-SAME: -> tensor<1x8x4x16xui8>

// -----

// Test 3: No requantization - DQ and Q use the same scale/zp (should not modify)
func.func @test_no_qdq_requantization(%arg0: tensor<1x8x64xf32>) -> tensor<1x8x16x4xui8> {
  %scale = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>

  %0 = "onnx.QuantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xui8>
  %1 = "onnx.DequantizeLinear"(%0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %2 = "onnx.Reshape"(%1, %shape) {allowzero = 0 : si64} : (tensor<1x8x64xf32>, tensor<4xi64>) -> tensor<1x8x16x4xf32>
  %3 = "onnx.QuantizeLinear"(%2, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x16x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x16x4xui8>

  return %3 : tensor<1x8x16x4xui8>
}

// CHECK-LABEL: func.func @test_no_qdq_requantization
// CHECK-DAG: %[[SCALE:.*]] = onnx.Constant dense<1.000000e-01>
// CHECK-DAG: %[[ZP:.*]] = onnx.Constant dense<0>
// Same scale/zp everywhere - no modification
// CHECK: %[[Q_PARENT:.*]] = "onnx.QuantizeLinear"(%arg0, %[[SCALE]], %[[ZP]])
// CHECK-SAME: -> tensor<1x8x64xui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q_PARENT]], %[[SCALE]], %[[ZP]])
// CHECK-SAME: -> tensor<1x8x64xf32>
// CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[DQ]]
// CHECK-SAME: -> tensor<1x8x16x4xf32>
// CHECK: %[[Q_OUT:.*]] = "onnx.QuantizeLinear"(%[[RESHAPE]], %[[SCALE]], %[[ZP]])
// CHECK-SAME: -> tensor<1x8x16x4xui8>

// -----

// Test 4: Slice with requantization
func.func @test_slice_qdq_requantization(%arg0: tensor<1x8x64xf32>) -> tensor<1x4x64xui8> {
  %scale1 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %scale2 = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %zp2 = onnx.Constant dense<5> : tensor<ui8>
  %starts = onnx.Constant dense<[0, 0, 0]> : tensor<3xi64>
  %ends = onnx.Constant dense<[1, 4, 64]> : tensor<3xi64>
  %axes = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
  %steps = onnx.Constant dense<[1, 1, 1]> : tensor<3xi64>

  %0 = "onnx.QuantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xui8>
  %1 = "onnx.DequantizeLinear"(%0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %2 = "onnx.Slice"(%1, %starts, %ends, %axes, %steps) : (tensor<1x8x64xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x64xf32>
  %3 = "onnx.QuantizeLinear"(%2, %scale2, %zp2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4x64xui8>

  return %3 : tensor<1x4x64xui8>
}

// CHECK-LABEL: func.func @test_slice_qdq_requantization
// CHECK-DAG: %[[S1:.*]] = onnx.Constant dense<1.000000e-01>
// CHECK-DAG: %[[ZP1:.*]] = onnx.Constant dense<0> : tensor<ui8>
// CHECK-DAG: %[[S2:.*]] = onnx.Constant dense<2.000000e-01>
// CHECK-DAG: %[[ZP2:.*]] = onnx.Constant dense<5>
// Q(parent) unchanged
// CHECK: %[[Q_PARENT:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S1]], %[[ZP1]])
// CHECK-SAME: -> tensor<1x8x64xui8>
// DQ updated to use s2/zp2
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q_PARENT]], %[[S2]], %[[ZP2]])
// CHECK-SAME: -> tensor<1x8x64xf32>
// CHECK: %[[SLICE:.*]] = "onnx.Slice"(%[[DQ]]
// CHECK-SAME: -> tensor<1x4x64xf32>
// CHECK: %[[Q_OUT:.*]] = "onnx.QuantizeLinear"(%[[SLICE]], %[[S2]], %[[ZP2]])
// CHECK-SAME: -> tensor<1x4x64xui8>

// -----

// Test 5: Concat with requantization - DQ nodes for each input should be updated
func.func @test_concat_qdq_requantization(%arg0: tensor<1x8x64xf32>, %arg1: tensor<1x8x64xf32>) -> tensor<1x16x64xui8> {
  %scale1 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %scale2 = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %zp2 = onnx.Constant dense<5> : tensor<ui8>

  %q0 = "onnx.QuantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xui8>
  %dq0 = "onnx.DequantizeLinear"(%q0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  %q1 = "onnx.QuantizeLinear"(%arg1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xui8>
  %dq1 = "onnx.DequantizeLinear"(%q1, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>

  %concat = "onnx.Concat"(%dq0, %dq1) {axis = 1 : si64} : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x16x64xf32>
  %out = "onnx.QuantizeLinear"(%concat, %scale2, %zp2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x16x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x16x64xui8>

  return %out : tensor<1x16x64xui8>
}

// CHECK-LABEL: func.func @test_concat_qdq_requantization
// CHECK-DAG: %[[S2:.*]] = onnx.Constant dense<2.000000e-01>
// CHECK-DAG: %[[ZP2:.*]] = onnx.Constant dense<5>
// Both DQ nodes updated to use s2/zp2
// CHECK: "onnx.DequantizeLinear"(%{{.*}}, %[[S2]], %[[ZP2]])
// CHECK-SAME: -> tensor<1x8x64xf32>
// CHECK: "onnx.DequantizeLinear"(%{{.*}}, %[[S2]], %[[ZP2]])
// CHECK-SAME: -> tensor<1x8x64xf32>
// CHECK: %[[CONCAT:.*]] = "onnx.Concat"
// CHECK-SAME: -> tensor<1x16x64xf32>
// CHECK: "onnx.QuantizeLinear"(%[[CONCAT]], %[[S2]], %[[ZP2]])
// CHECK-SAME: -> tensor<1x16x64xui8>

// -----

// Test 6: DQ has multiple uses - should not modify
func.func @test_dq_multiple_uses(%arg0: tensor<1x8x64xf32>) -> (tensor<1x8x16x4xui8>, tensor<1x8x64xf32>) {
  %scale1 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp1 = onnx.Constant dense<0> : tensor<ui8>
  %scale2 = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %zp2 = onnx.Constant dense<5> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 8, 16, 4]> : tensor<4xi64>

  %0 = "onnx.QuantizeLinear"(%arg0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xui8>
  %1 = "onnx.DequantizeLinear"(%0, %scale1, %zp1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x8x64xf32>
  %2 = "onnx.Reshape"(%1, %shape) {allowzero = 0 : si64} : (tensor<1x8x64xf32>, tensor<4xi64>) -> tensor<1x8x16x4xf32>
  %3 = "onnx.QuantizeLinear"(%2, %scale2, %zp2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x8x16x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x8x16x4xui8>

  // %1 (DQ output) has two uses: Reshape and return
  return %3, %1 : tensor<1x8x16x4xui8>, tensor<1x8x64xf32>
}

// CHECK-LABEL: func.func @test_dq_multiple_uses
// CHECK-DAG: %[[S1:.*]] = onnx.Constant dense<1.000000e-01>
// CHECK-DAG: %[[ZP1:.*]] = onnx.Constant dense<0> : tensor<ui8>
// DQ should NOT be modified (multiple uses)
// CHECK: %[[Q_PARENT:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S1]], %[[ZP1]])
// CHECK-SAME: -> tensor<1x8x64xui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q_PARENT]], %[[S1]], %[[ZP1]])
// CHECK-SAME: -> tensor<1x8x64xf32>
// CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[DQ]]
// CHECK-SAME: -> tensor<1x8x16x4xf32>
// CHECK: "onnx.QuantizeLinear"(%[[RESHAPE]]
// CHECK-SAME: -> tensor<1x8x16x4xui8>
