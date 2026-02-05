// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// RUN: onnx-mlir-opt --split-input-file --replace-adjacent-ops %s | FileCheck %s

// Test 1: Merge nested concat operations on same axis.
func.func @test_merge_nested_concat(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>,
                                     %arg2: tensor<1x2x3x4xf32>, %arg3: tensor<1x2x3x4xf32>,
                                     %arg4: tensor<1x2x3x4xf32>) -> tensor<1x2x3x20xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x8xf32>
  %1 = "onnx.Concat"(%arg2, %arg3) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x8xf32>
  %2 = "onnx.Concat"(%0, %1, %arg4) {axis = 3 : si64} : (tensor<1x2x3x8xf32>, tensor<1x2x3x8xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x20xf32>
  return %2 : tensor<1x2x3x20xf32>
}
// CHECK: %[[MERGED:.*]] = "onnx.Concat"(%arg0, %arg1, %arg2, %arg3, %arg4) {axis = 3 : si64}
// CHECK-SAME: (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x20xf32>
// CHECK: return %[[MERGED]]

// -----

// Test 2: Don't merge concat operations on different axes.
func.func @test_no_merge_different_axis(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>,
                                         %arg2: tensor<1x2x6x4xf32>) -> tensor<1x2x6x8xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 2 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x6x4xf32>
  %1 = "onnx.Concat"(%0, %arg2) {axis = 3 : si64} : (tensor<1x2x6x4xf32>, tensor<1x2x6x4xf32>) -> tensor<1x2x6x8xf32>
  return %1 : tensor<1x2x6x8xf32>
}
// CHECK: %[[CONCAT1:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 2 : si64}
// CHECK: %[[CONCAT2:.*]] = "onnx.Concat"(%[[CONCAT1]], %arg2) {axis = 3 : si64}

// -----

// Test 3: Split operation with duplicate inputs.
// CHECK-LABEL: func.func @test_split_duplicate_inputs
func.func @test_split_duplicate_inputs(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x12xf32> {
  %0 = "onnx.Concat"(%arg0, %arg0, %arg0) {axis = 3 : si64} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x12xf32>
  return %0 : tensor<1x2x3x12xf32>
}
// Pass inserts onnx.Shape + onnx.Reshape (dynamic-safe) for duplicates.
// Order of inserted reshapes is not guaranteed; use DAG.
// CHECK-DAG: %[[S0:.*]] = "onnx.Shape"(%arg0) {start = 0 : si64} : (tensor<1x2x3x4xf32>) -> tensor<?xi64>
// CHECK-DAG: %[[R0:.*]] = "onnx.Reshape"(%arg0, %[[S0]]) {allowzero = 0 : si64, duplicate_input = true} : (tensor<1x2x3x4xf32>, tensor<?xi64>) -> tensor<1x2x3x4xf32>
// CHECK-DAG: %[[S1:.*]] = "onnx.Shape"(%arg0) {start = 0 : si64} : (tensor<1x2x3x4xf32>) -> tensor<?xi64>
// CHECK-DAG: %[[R1:.*]] = "onnx.Reshape"(%arg0, %[[S1]]) {allowzero = 0 : si64, duplicate_input = true} : (tensor<1x2x3x4xf32>, tensor<?xi64>) -> tensor<1x2x3x4xf32>
// CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%arg0, %{{.*}}, %{{.*}}) {axis = 3 : si64}

// -----

// Test 4: Split Add operation with duplicate inputs.
// CHECK-LABEL: func.func @test_split_duplicate_add
func.func @test_split_duplicate_add(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  %0 = "onnx.Add"(%arg0, %arg0) : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}
// CHECK: %[[S:.*]] = "onnx.Shape"(%arg0) {start = 0 : si64} : (tensor<1x2x3x4xf32>) -> tensor<?xi64>
// CHECK: %[[R:.*]] = "onnx.Reshape"(%arg0, %[[S]]) {allowzero = 0 : si64, duplicate_input = true} : (tensor<1x2x3x4xf32>, tensor<?xi64>) -> tensor<1x2x3x4xf32>
// CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %[[R]]) : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>

