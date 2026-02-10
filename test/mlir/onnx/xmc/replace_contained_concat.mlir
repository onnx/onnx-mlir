// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// RUN: onnx-mlir-opt --split-input-file --replace-contained-concat %s | FileCheck %s

func.func @test_contained_concat_basic(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>, %arg2: tensor<1x2x3x4xf32>, %arg3: tensor<1x2x3x4xf32>, %arg4: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x12xf32>, tensor<1x2x3x20xf32>) {
  %0 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x12xf32>
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2, %arg3, %arg4) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x20xf32>
  return %0, %1 : tensor<1x2x3x12xf32>, tensor<1x2x3x20xf32>
}

// CHECK: %[[INNER:.*]] = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x12xf32>
// CHECK: %[[OUTER:.*]] = "onnx.Concat"(%[[INNER]], %arg3, %arg4) {axis = 3 : si64} : (tensor<1x2x3x12xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x20xf32>
// CHECK: return %[[INNER]], %[[OUTER]]

// -----

// Test 2: No optimization if axes differ.
// CHECK-LABEL: func.func @test_different_axes
func.func @test_different_axes(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>, %arg2: tensor<1x2x3x4xf32>, %arg3: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x8xf32>, tensor<1x2x9x4xf32>) {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x8xf32>
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2, %arg3) {axis = 2 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x9x4xf32>
  return %0, %1 : tensor<1x2x3x8xf32>, tensor<1x2x9x4xf32>
}
// CHECK: %[[C0:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64}
// CHECK: %[[C1:.*]] = "onnx.Concat"(%arg0, %arg1, %arg2, %arg3) {axis = 2 : si64}
// CHECK: return %[[C0]], %[[C1]]

// -----

// Test 3: No optimization if not a subset.
// concat1([A, B]) and concat2([B, C, D]) - A is not in concat2's inputs.
// CHECK-LABEL: func.func @test_not_subset
func.func @test_not_subset(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>, %arg2: tensor<1x2x3x4xf32>, %arg3: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x8xf32>, tensor<1x2x3x12xf32>) {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x8xf32>
  %1 = "onnx.Concat"(%arg1, %arg2, %arg3) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x12xf32>
  return %0, %1 : tensor<1x2x3x8xf32>, tensor<1x2x3x12xf32>
}
// CHECK: %[[C0:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64}
// CHECK: %[[C1:.*]] = "onnx.Concat"(%arg1, %arg2, %arg3) {axis = 3 : si64}
// CHECK: return %[[C0]], %[[C1]]

// -----

// Test 4: Subset exists but order differs; still optimize by reusing inner.
// inner: [A, B], outer: [B, A, C] (subset true, order differs)
// CHECK-LABEL: func.func @test_subset_wrong_order
func.func @test_subset_wrong_order(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>, %arg2: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x8xf32>, tensor<1x2x3x12xf32>) {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x8xf32>
  %1 = "onnx.Concat"(%arg1, %arg0, %arg2) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x12xf32>
  return %0, %1 : tensor<1x2x3x8xf32>, tensor<1x2x3x12xf32>
}
// CHECK: %[[I0:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64}
// CHECK: %[[O0:.*]] = "onnx.Concat"(%[[I0]], %arg2) {axis = 3 : si64}
// CHECK: return %[[I0]], %[[O0]]

// -----

// Test 5: Quantized concat with matching parameters.
// CHECK-LABEL: func.func @test_quantized_matching
func.func @test_quantized_matching(%arg0: tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>, %arg1: tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>, %arg2: tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>, %arg3: tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>) -> (tensor<1x2x3x8x!quant.uniform<u8:f32, 0.5:128>>, tensor<1x2x3x16x!quant.uniform<u8:f32, 0.5:128>>) {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64} : (tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>, tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x2x3x8x!quant.uniform<u8:f32, 0.5:128>>
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2, %arg3) {axis = 3 : si64} : (tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>, tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>, tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>, tensor<1x2x3x4x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x2x3x16x!quant.uniform<u8:f32, 0.5:128>>
  return %0, %1 : tensor<1x2x3x8x!quant.uniform<u8:f32, 0.5:128>>, tensor<1x2x3x16x!quant.uniform<u8:f32, 0.5:128>>
}
// CHECK: %[[INNER:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64}
// CHECK: %[[OUTER:.*]] = "onnx.Concat"(%[[INNER]], %arg2, %arg3) {axis = 3 : si64}
// CHECK: return %[[INNER]], %[[OUTER]]

// -----

// Test 6: Chained optimization.
// CHECK-LABEL: func.func @test_chained_optimization
func.func @test_chained_optimization(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>, %arg2: tensor<1x2x3x4xf32>, %arg3: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x8xf32>, tensor<1x2x3x12xf32>, tensor<1x2x3x16xf32>) {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x8xf32>
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x12xf32>
  %2 = "onnx.Concat"(%arg0, %arg1, %arg2, %arg3) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x16xf32>
  return %0, %1, %2 : tensor<1x2x3x8xf32>, tensor<1x2x3x12xf32>, tensor<1x2x3x16xf32>
}
// CHECK: %[[C1:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64}
// CHECK: %[[C2:.*]] = "onnx.Concat"(%[[C1]], %arg2) {axis = 3 : si64}
// CHECK: %[[C3:.*]] = "onnx.Concat"(%[[C2]], %arg3) {axis = 3 : si64}
// CHECK: return %[[C1]], %[[C2]], %[[C3]]

