// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --optimize-slice-reshape-transpose-block %s | FileCheck %s

// Covers OptimizeSliceReshapeTransposeBlockPass: rank-3 reshape inputs, same 4D reshape
// on each branch, head-split transposes {0,2,1,3} and/or {0,2,3,1}. Branch order is
// canonicalized using slice starts[2] (must be onnx.Constant). MHA: all {0,2,1,3}.
// Matmul template: {0,2,1,3}, {0,2,3,1}, {0,2,1,3}; middle output gets extra
// Transpose {0,1,3,2} after slicing.

module {
  // Test 1: MHA — three chains all use perm [0, 2, 1, 3]; no post-slice transpose.
  // Input shape: [16, 225, 576] -> 3 slices of [16, 225, 192] each
  // Each slice -> reshape [16, 225, 12, 16] -> transpose {0, 2, 1, 3} -> [16, 12, 225, 16]
  // Optimized: reshape [16,225,36,16] -> transpose {0,2,1,3} -> three slices on fused H.
  // CHECK-LABEL: func.func @mha_slice_reshape_transpose
  func.func @mha_slice_reshape_transpose(%arg0: tensor<16x225x576xf32>) -> (tensor<16x12x225x16xf32>, tensor<16x12x225x16xf32>, tensor<16x12x225x16xf32>) {
    // Slice parameters for Q, K, V
    %starts0 = "onnx.Constant"() {value = dense<[0, 0, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ends0 = "onnx.Constant"() {value = dense<[16, 225, 192]> : tensor<3xi64>} : () -> tensor<3xi64>
    %starts1 = "onnx.Constant"() {value = dense<[0, 0, 192]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ends1 = "onnx.Constant"() {value = dense<[16, 225, 384]> : tensor<3xi64>} : () -> tensor<3xi64>
    %starts2 = "onnx.Constant"() {value = dense<[0, 0, 384]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ends2 = "onnx.Constant"() {value = dense<[16, 225, 576]> : tensor<3xi64>} : () -> tensor<3xi64>
    %axes = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
    %steps = "onnx.Constant"() {value = dense<[1, 1, 1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %reshape_shape = "onnx.Constant"() {value = dense<[16, 225, 12, 16]> : tensor<4xi64>} : () -> tensor<4xi64>

    // Chain 0: slice -> reshape -> transpose (Q)
    %slice0 = "onnx.Slice"(%arg0, %starts0, %ends0, %axes, %steps) : (tensor<16x225x576xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<16x225x192xf32>
    %reshape0 = "onnx.Reshape"(%slice0, %reshape_shape) {allowzero = 0 : si64} : (tensor<16x225x192xf32>, tensor<4xi64>) -> tensor<16x225x12x16xf32>
    %transpose0 = "onnx.Transpose"(%reshape0) {perm = [0, 2, 1, 3]} : (tensor<16x225x12x16xf32>) -> tensor<16x12x225x16xf32>

    // Chain 1: slice -> reshape -> transpose (K)
    %slice1 = "onnx.Slice"(%arg0, %starts1, %ends1, %axes, %steps) : (tensor<16x225x576xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<16x225x192xf32>
    %reshape1 = "onnx.Reshape"(%slice1, %reshape_shape) {allowzero = 0 : si64} : (tensor<16x225x192xf32>, tensor<4xi64>) -> tensor<16x225x12x16xf32>
    %transpose1 = "onnx.Transpose"(%reshape1) {perm = [0, 2, 1, 3]} : (tensor<16x225x12x16xf32>) -> tensor<16x12x225x16xf32>

    // Chain 2: slice -> reshape -> transpose (V)
    %slice2 = "onnx.Slice"(%arg0, %starts2, %ends2, %axes, %steps) : (tensor<16x225x576xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<16x225x192xf32>
    %reshape2 = "onnx.Reshape"(%slice2, %reshape_shape) {allowzero = 0 : si64} : (tensor<16x225x192xf32>, tensor<4xi64>) -> tensor<16x225x12x16xf32>
    %transpose2 = "onnx.Transpose"(%reshape2) {perm = [0, 2, 1, 3]} : (tensor<16x225x12x16xf32>) -> tensor<16x12x225x16xf32>

    return %transpose0, %transpose1, %transpose2 : tensor<16x12x225x16xf32>, tensor<16x12x225x16xf32>, tensor<16x12x225x16xf32>
  }
  // Single reshape -> single transpose -> 3 slices (instead of 3x slice->reshape->transpose)
  // CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%arg0
  // CHECK-SAME: -> tensor<16x225x36x16xf32>
  // CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[RESHAPE]]) {perm = [0, 2, 1, 3]}
  // CHECK: "onnx.Slice"(%[[TRANSPOSE]]
  // CHECK: "onnx.Slice"(%[[TRANSPOSE]]
  // CHECK: "onnx.Slice"(%[[TRANSPOSE]]
  // CHECK-NOT: {perm = [0, 1, 3, 2]}
  // CHECK-LABEL: func.func @matmul_mixed_transpose_qkv

  // Test 2: Matmul-style — middle branch perm [0, 2, 3, 1] -> extra Transpose [0, 1, 3, 2] after its slice.
  func.func @matmul_mixed_transpose_qkv(%arg0: tensor<16x225x576xf32>) -> (tensor<16x12x225x16xf32>, tensor<16x12x16x225xf32>, tensor<16x12x225x16xf32>) {
    %starts0 = "onnx.Constant"() {value = dense<[0, 0, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ends0 = "onnx.Constant"() {value = dense<[16, 225, 192]> : tensor<3xi64>} : () -> tensor<3xi64>
    %starts1 = "onnx.Constant"() {value = dense<[0, 0, 192]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ends1 = "onnx.Constant"() {value = dense<[16, 225, 384]> : tensor<3xi64>} : () -> tensor<3xi64>
    %starts2 = "onnx.Constant"() {value = dense<[0, 0, 384]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ends2 = "onnx.Constant"() {value = dense<[16, 225, 576]> : tensor<3xi64>} : () -> tensor<3xi64>
    %axes = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
    %steps = "onnx.Constant"() {value = dense<[1, 1, 1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %reshape_shape = "onnx.Constant"() {value = dense<[16, 225, 12, 16]> : tensor<4xi64>} : () -> tensor<4xi64>

    %slice0 = "onnx.Slice"(%arg0, %starts0, %ends0, %axes, %steps) : (tensor<16x225x576xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<16x225x192xf32>
    %reshape0 = "onnx.Reshape"(%slice0, %reshape_shape) {allowzero = 0 : si64} : (tensor<16x225x192xf32>, tensor<4xi64>) -> tensor<16x225x12x16xf32>
    %transpose0 = "onnx.Transpose"(%reshape0) {perm = [0, 2, 1, 3]} : (tensor<16x225x12x16xf32>) -> tensor<16x12x225x16xf32>

    %slice1 = "onnx.Slice"(%arg0, %starts1, %ends1, %axes, %steps) : (tensor<16x225x576xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<16x225x192xf32>
    %reshape1 = "onnx.Reshape"(%slice1, %reshape_shape) {allowzero = 0 : si64} : (tensor<16x225x192xf32>, tensor<4xi64>) -> tensor<16x225x12x16xf32>
    %transpose1 = "onnx.Transpose"(%reshape1) {perm = [0, 2, 3, 1]} : (tensor<16x225x12x16xf32>) -> tensor<16x12x16x225xf32>

    %slice2 = "onnx.Slice"(%arg0, %starts2, %ends2, %axes, %steps) : (tensor<16x225x576xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<16x225x192xf32>
    %reshape2 = "onnx.Reshape"(%slice2, %reshape_shape) {allowzero = 0 : si64} : (tensor<16x225x192xf32>, tensor<4xi64>) -> tensor<16x225x12x16xf32>
    %transpose2 = "onnx.Transpose"(%reshape2) {perm = [0, 2, 1, 3]} : (tensor<16x225x12x16xf32>) -> tensor<16x12x225x16xf32>

    return %transpose0, %transpose1, %transpose2 : tensor<16x12x225x16xf32>, tensor<16x12x16x225xf32>, tensor<16x12x225x16xf32>
  }
  // Common fused reshape/transpose; exactly one fixup transpose on the middle path.
  // CHECK: %[[RESHAPE_M:.*]] = "onnx.Reshape"(%arg0
  // CHECK-SAME: -> tensor<16x225x36x16xf32>
  // CHECK: %[[TRANSPOSE_M:.*]] = "onnx.Transpose"(%[[RESHAPE_M]]) {perm = [0, 2, 1, 3]}
  // CHECK: "onnx.Slice"(%[[TRANSPOSE_M]]
  // CHECK: "onnx.Slice"(%[[TRANSPOSE_M]]
  // CHECK: %{{.+}} = "onnx.Transpose"({{.+}}) {perm = [0, 1, 3, 2]}
  // CHECK: "onnx.Slice"(%[[TRANSPOSE_M]]

  // Test 3: MHA with quantized element type (same as Test 1).
  // CHECK-LABEL: func.func @mha_slice_reshape_transpose_quant
  func.func @mha_slice_reshape_transpose_quant(%arg0: tensor<16x225x576x!quant.uniform<i8:f32, 0.1:0>>) -> (tensor<16x12x225x16x!quant.uniform<i8:f32, 0.1:0>>, tensor<16x12x225x16x!quant.uniform<i8:f32, 0.1:0>>, tensor<16x12x225x16x!quant.uniform<i8:f32, 0.1:0>>) {
    // Slice parameters for Q, K, V
    %starts0 = "onnx.Constant"() {value = dense<[0, 0, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ends0 = "onnx.Constant"() {value = dense<[16, 225, 192]> : tensor<3xi64>} : () -> tensor<3xi64>
    %starts1 = "onnx.Constant"() {value = dense<[0, 0, 192]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ends1 = "onnx.Constant"() {value = dense<[16, 225, 384]> : tensor<3xi64>} : () -> tensor<3xi64>
    %starts2 = "onnx.Constant"() {value = dense<[0, 0, 384]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ends2 = "onnx.Constant"() {value = dense<[16, 225, 576]> : tensor<3xi64>} : () -> tensor<3xi64>
    %axes = "onnx.Constant"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
    %steps = "onnx.Constant"() {value = dense<[1, 1, 1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %reshape_shape = "onnx.Constant"() {value = dense<[16, 225, 12, 16]> : tensor<4xi64>} : () -> tensor<4xi64>

    // Chain 0: slice -> reshape -> transpose (Q)
    %slice0 = "onnx.Slice"(%arg0, %starts0, %ends0, %axes, %steps) : (tensor<16x225x576x!quant.uniform<i8:f32, 0.1:0>>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<16x225x192x!quant.uniform<i8:f32, 0.1:0>>
    %reshape0 = "onnx.Reshape"(%slice0, %reshape_shape) {allowzero = 0 : si64} : (tensor<16x225x192x!quant.uniform<i8:f32, 0.1:0>>, tensor<4xi64>) -> tensor<16x225x12x16x!quant.uniform<i8:f32, 0.1:0>>
    %transpose0 = "onnx.Transpose"(%reshape0) {perm = [0, 2, 1, 3]} : (tensor<16x225x12x16x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<16x12x225x16x!quant.uniform<i8:f32, 0.1:0>>

    // Chain 1: slice -> reshape -> transpose (K)
    %slice1 = "onnx.Slice"(%arg0, %starts1, %ends1, %axes, %steps) : (tensor<16x225x576x!quant.uniform<i8:f32, 0.1:0>>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<16x225x192x!quant.uniform<i8:f32, 0.1:0>>
    %reshape1 = "onnx.Reshape"(%slice1, %reshape_shape) {allowzero = 0 : si64} : (tensor<16x225x192x!quant.uniform<i8:f32, 0.1:0>>, tensor<4xi64>) -> tensor<16x225x12x16x!quant.uniform<i8:f32, 0.1:0>>
    %transpose1 = "onnx.Transpose"(%reshape1) {perm = [0, 2, 1, 3]} : (tensor<16x225x12x16x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<16x12x225x16x!quant.uniform<i8:f32, 0.1:0>>

    // Chain 2: slice -> reshape -> transpose (V)
    %slice2 = "onnx.Slice"(%arg0, %starts2, %ends2, %axes, %steps) : (tensor<16x225x576x!quant.uniform<i8:f32, 0.1:0>>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<16x225x192x!quant.uniform<i8:f32, 0.1:0>>
    %reshape2 = "onnx.Reshape"(%slice2, %reshape_shape) {allowzero = 0 : si64} : (tensor<16x225x192x!quant.uniform<i8:f32, 0.1:0>>, tensor<4xi64>) -> tensor<16x225x12x16x!quant.uniform<i8:f32, 0.1:0>>
    %transpose2 = "onnx.Transpose"(%reshape2) {perm = [0, 2, 1, 3]} : (tensor<16x225x12x16x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<16x12x225x16x!quant.uniform<i8:f32, 0.1:0>>

    return %transpose0, %transpose1, %transpose2 : tensor<16x12x225x16x!quant.uniform<i8:f32, 0.1:0>>, tensor<16x12x225x16x!quant.uniform<i8:f32, 0.1:0>>, tensor<16x12x225x16x!quant.uniform<i8:f32, 0.1:0>>
  }
  // Quantized: Single reshape -> single transpose -> 3 slices, element type preserved
  // CHECK: %[[RESHAPE_Q:.*]] = "onnx.Reshape"(%arg0
  // CHECK-SAME: -> tensor<16x225x36x16x!quant.uniform<i8:f32, 1.000000e-01>>
  // CHECK: %[[TRANSPOSE_Q:.*]] = "onnx.Transpose"(%[[RESHAPE_Q]]) {perm = [0, 2, 1, 3]}
  // CHECK: "onnx.Slice"(%[[TRANSPOSE_Q]]
  // CHECK: "onnx.Slice"(%[[TRANSPOSE_Q]]
  // CHECK: "onnx.Slice"(%[[TRANSPOSE_Q]]
}

