// RUN: onnx-mlir-opt --split-input-file --combine-transpose-pair %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

// Test case: Two transpose ops with same input and same perm should be combined
// CHECK-LABEL: @combine_transpose_pair_basic
func.func @combine_transpose_pair_basic(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>) {
    // Two transposes with identical input and perm
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    return %0, %1 : tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>
}

// Verify only one transpose remains
// CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]}
// CHECK-NOT: "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]}
// CHECK: return %[[TRANSPOSE]], %[[TRANSPOSE]]


// Test case: Transposes with different perms should NOT be combined
// CHECK-LABEL: @no_combine_different_perm
func.func @no_combine_different_perm(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x3x2x4xf32>, tensor<1x4x3x2xf32>) {
    // Two transposes with same input but different perm - should NOT be combined
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 3, 2, 1]} : (tensor<1x2x3x4xf32>) -> tensor<1x4x3x2xf32>
    return %0, %1 : tensor<1x3x2x4xf32>, tensor<1x4x3x2xf32>
}

// Both transposes should remain since they have different perms
// CHECK: %[[T0:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]}
// CHECK: %[[T1:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 3, 2, 1]}
// CHECK: return %[[T0]], %[[T1]]


// Test case: Transposes with different inputs should NOT be combined
// CHECK-LABEL: @no_combine_different_input
func.func @no_combine_different_input(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> (tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>) {
    // Two transposes with same perm but different inputs - should NOT be combined
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    return %0, %1 : tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>
}

// Both transposes should remain since they have different inputs
// CHECK: %[[T0:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]}
// CHECK: %[[T1:.*]] = "onnx.Transpose"(%arg1) {perm = [0, 2, 1, 3]}
// CHECK: return %[[T0]], %[[T1]]


// Test case: Multiple duplicate pairs - each pair should be combined separately
// CHECK-LABEL: @combine_multiple_pairs
func.func @combine_multiple_pairs(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>, tensor<1x4x3x2xf32>, tensor<1x4x3x2xf32>) {
    // First pair with perm [0, 2, 1, 3]
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    // Second pair with perm [0, 3, 2, 1]
    %2 = "onnx.Transpose"(%arg0) {perm = [0, 3, 2, 1]} : (tensor<1x2x3x4xf32>) -> tensor<1x4x3x2xf32>
    %3 = "onnx.Transpose"(%arg0) {perm = [0, 3, 2, 1]} : (tensor<1x2x3x4xf32>) -> tensor<1x4x3x2xf32>
    return %0, %1, %2, %3 : tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>, tensor<1x4x3x2xf32>, tensor<1x4x3x2xf32>
}

// Verify each pair is combined to one transpose
// CHECK: %[[T1:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]}
// CHECK: %[[T2:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 3, 2, 1]}
// CHECK-NOT: "onnx.Transpose"
// CHECK: return %[[T1]], %[[T1]], %[[T2]], %[[T2]]

// Test case: 4 transpose
// CHECK-LABEL: @combine_4_transpose
func.func @combine_4_transpose(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>) {
    // First pair with perm [0, 2, 1, 3]
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    %2 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    %3 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    return %0, %1, %2, %3 : tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>
}

// Verify all are combined to one transpose
// CHECK: %[[T1:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]}
// CHECK-NOT: "onnx.Transpose"
// CHECK: return %[[T1]], %[[T1]], %[[T1]], %[[T1]]

// Test case: 2 transpose with an op in between
// CHECK-LABEL: @order_of_transpose
func.func @order_of_transpose(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>, tensor<6x4xf32>) {
    %t1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>  // earlier
    %shape = onnx.Constant dense<[6, 4]> : tensor<2xi64>
    %reshaped = "onnx.Reshape"(%t1, %shape) {allowzero = 0 : si64} : (tensor<1x3x2x4xf32>, tensor<2xi64>) -> tensor<6x4xf32>
    %t2 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>  // later
    return %t1, %t2, %reshaped : tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>, tensor<6x4xf32>
}

// The reshape still uses the result from transpose
// CHECK: %[[SHAPE:.*]] = onnx.Constant dense<[6, 4]>
// CHECK: %[[T2:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]}
// CHECK: %[[RESHAPED:.*]] = "onnx.Reshape"(%[[T2]], %[[SHAPE]])
// CHECK-NOT: "onnx.Transpose"
// CHECK: return %[[T2]], %[[T2]], %[[RESHAPED]]
