// RUN: onnx-mlir-opt --split-input-file --transfer-softmax-axis-to-last %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// CHECK-LABEL: @softmax_axis_1_rank4
func.func @softmax_axis_1_rank4(%arg0: tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32> {
  // CHECK: %[[TIN:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x16x4x8400xf32>) -> tensor<1x4x8400x16xf32>
  // CHECK: %[[S:.*]] = "onnx.Softmax"(%[[TIN]]) {axis = -1 : si64}
  // CHECK-SAME: (tensor<1x4x8400x16xf32>) -> tensor<1x4x8400x16xf32>
  // CHECK: %[[TOUT:.*]] = "onnx.Transpose"(%[[S]]) {perm = [0, 3, 1, 2]}
  // CHECK-SAME: (tensor<1x4x8400x16xf32>) -> tensor<1x16x4x8400xf32>
  %0 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32>
  return %0 : tensor<1x16x4x8400xf32>
}

// -----

// CHECK-LABEL: @softmax_axis_2_rank4
func.func @softmax_axis_2_rank4(%arg0: tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32> {
  // CHECK: %[[TIN:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 1, 3, 2]}
  // CHECK-SAME: (tensor<1x16x4x8400xf32>) -> tensor<1x16x8400x4xf32>
  // CHECK: %[[S:.*]] = "onnx.Softmax"(%[[TIN]]) {axis = -1 : si64}
  // CHECK: %[[TOUT:.*]] = "onnx.Transpose"(%[[S]]) {perm = [0, 1, 3, 2]}
  %0 = "onnx.Softmax"(%arg0) {axis = 2 : si64} : (tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32>
  return %0 : tensor<1x16x4x8400xf32>
}

// -----

// CHECK-LABEL: @softmax_axis_1_rank3
func.func @softmax_axis_1_rank3(%arg0: tensor<1x16x8400xf32>) -> tensor<1x16x8400xf32> {
  // CHECK: %[[TIN:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]}
  // CHECK-SAME: (tensor<1x16x8400xf32>) -> tensor<1x8400x16xf32>
  // CHECK: %[[S:.*]] = "onnx.Softmax"(%[[TIN]]) {axis = -1 : si64}
  // CHECK: %[[TOUT:.*]] = "onnx.Transpose"(%[[S]]) {perm = [0, 2, 1]}
  %0 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<1x16x8400xf32>) -> tensor<1x16x8400xf32>
  return %0 : tensor<1x16x8400xf32>
}

// -----

// CHECK-LABEL: @softmax_axis_1_quant
func.func @softmax_axis_1_quant(%arg0: tensor<1x16x4x8400x!quant.uniform<i16:f32, 6.16E-4>>) -> tensor<1x16x4x8400x!quant.uniform<i16:f32, 3.05E-5>> {
  // CHECK: %[[TIN:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK: %[[S:.*]] = "onnx.Softmax"(%[[TIN]]) {axis = -1 : si64}
  // CHECK: %[[TOUT:.*]] = "onnx.Transpose"(%[[S]]) {perm = [0, 3, 1, 2]}
  %0 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<1x16x4x8400x!quant.uniform<i16:f32, 6.16E-4>>) -> tensor<1x16x4x8400x!quant.uniform<i16:f32, 3.05E-5>>
  return %0 : tensor<1x16x4x8400x!quant.uniform<i16:f32, 3.05E-5>>
}

// -----

// CHECK-LABEL: @softmax_axis_last_positive
func.func @softmax_axis_last_positive(%arg0: tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32> {
  // CHECK-NOT: "onnx.Transpose"
  // CHECK: %[[S:.*]] = "onnx.Softmax"(%arg0) {axis = 3 : si64}
  // CHECK: return %[[S]]
  %0 = "onnx.Softmax"(%arg0) {axis = 3 : si64} : (tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32>
  return %0 : tensor<1x16x4x8400xf32>
}

// -----

// CHECK-LABEL: @softmax_axis_neg_one
func.func @softmax_axis_neg_one(%arg0: tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32> {
  // CHECK-NOT: "onnx.Transpose"
  // CHECK: %[[S:.*]] = "onnx.Softmax"(%arg0) {axis = -1 : si64}
  // CHECK: return %[[S]]
  %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64} : (tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32>
  return %0 : tensor<1x16x4x8400xf32>
}

// -----

// CHECK-LABEL: @softmax_followed_by_log_skipped
func.func @softmax_followed_by_log_skipped(%arg0: tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32> {
  // CHECK-NOT: "onnx.Transpose"
  // CHECK: %[[S:.*]] = "onnx.Softmax"(%arg0) {axis = 1 : si64}
  // CHECK: %[[L:.*]] = "onnx.Log"(%[[S]])
  // CHECK: return %[[L]]
  %0 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32>
  %1 = "onnx.Log"(%0) : (tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32>
  return %1 : tensor<1x16x4x8400xf32>
}

// -----

// CHECK-LABEL: @log_softmax_skipped
func.func @log_softmax_skipped(%arg0: tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32> {
  // CHECK-NOT: "onnx.Transpose"
  // CHECK: %[[LS:.*]] = "onnx.LogSoftmax"(%arg0) {axis = 1 : si64}
  // CHECK: return %[[LS]]
  %0 = "onnx.LogSoftmax"(%arg0) {axis = 1 : si64} : (tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32>
  return %0 : tensor<1x16x4x8400xf32>
}
