// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s
//
// Tests for ONNXSliceOp::inferShapes normalization of starts/ends/axes.
//
// Shape inference materializes None axes -> [0,...,n-1] and None steps -> [1,...,1]
// and then normalizes:
//   - axes to non-negative (wrap by rank)
//   - starts and ends (positive step only): wrap negatives by dim, clamp to [0, dim]
//
// Negative-step start/end normalization is intentionally skipped because the
// -1 "before-index-0" end sentinel is not idempotent under repeated inferShapes.

// -----
// INT64_MAX end clamped to dim (64); explicit step=1 and axis=3 preserved.
func.func @slice_clamps_int64_max_end(%arg0: tensor<1x1600x3x64xf32>) -> tensor<1x1600x3x32xf32> {
  %starts = onnx.Constant dense<32> : tensor<1xi64>
  %ends   = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %axes   = onnx.Constant dense<3> : tensor<1xi64>
  %steps  = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<1x1600x3x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1600x3x32xf32>
  return %0 : tensor<1x1600x3x32xf32>
// CHECK-LABEL:  func.func @slice_clamps_int64_max_end
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1600x3x64xf32>) -> tensor<1x1600x3x32xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<64> : tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_3_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<1x1600x3x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1600x3x32xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x1600x3x32xf32>
}

// -----
// Negative axis (-1 -> 2) and negative start (-2 -> 2) normalized.
func.func @slice_normalizes_negative_axis_and_start(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x2xf32> {
  %starts = onnx.Constant dense<-2> : tensor<1xi64>
  %ends   = onnx.Constant dense<4> : tensor<1xi64>
  %axes   = onnx.Constant dense<-1> : tensor<1xi64>
  %steps  = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3x2xf32>
  return %0 : tensor<2x3x2xf32>
// CHECK-LABEL:  func.func @slice_normalizes_negative_axis_and_start
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x3x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_2_]], [[VAR_0_]], [[VAR_2_]], [[VAR_1_]]) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3x2xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x3x2xf32>
}

// -----
// None axes materialized to [0]; None steps materialized to [1]; then
// starts/ends already canonical -> no further change.
func.func @slice_materializes_none_axes_and_steps(%arg0: tensor<8xf32>) -> tensor<3xf32> {
  %starts = onnx.Constant dense<2> : tensor<1xi64>
  %ends   = onnx.Constant dense<5> : tensor<1xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %none, %none) : (tensor<8xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func.func @slice_materializes_none_axes_and_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]]) : (tensor<8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// CHECK:           return [[VAR_4_]] : tensor<3xf32>
}

// -----
// Already canonical: no rewrites expected.
func.func @slice_already_canonical(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x2xf32> {
  %starts = onnx.Constant dense<2> : tensor<1xi64>
  %ends   = onnx.Constant dense<4> : tensor<1xi64>
  %axes   = onnx.Constant dense<2> : tensor<1xi64>
  %steps  = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3x2xf32>
  return %0 : tensor<2x3x2xf32>
// CHECK-LABEL: func.func @slice_already_canonical
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x3x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_0_]], [[VAR_2_]]) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3x2xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x3x2xf32>
}

// -----
// Overshoot end (100 > dim=6) clamped to 6.
func.func @slice_clamps_overshoot_end(%arg0: tensor<5x6xf32>) -> tensor<5x4xf32> {
  %starts = onnx.Constant dense<2> : tensor<1xi64>
  %ends   = onnx.Constant dense<100> : tensor<1xi64>
  %axes   = onnx.Constant dense<1> : tensor<1xi64>
  %steps  = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<5x6xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
// CHECK-LABEL:  func.func @slice_clamps_overshoot_end
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x6xf32>) -> tensor<5x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_2_]], [[VAR_1_]], [[VAR_1_]]) : (tensor<5x6xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5x4xf32>
// CHECK:           return [[VAR_3_]] : tensor<5x4xf32>
}

// -----
// Dynamic input: shape inference cannot normalize without static dims; bails.
func.func @slice_dynamic_data_unchanged(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %starts = onnx.Constant dense<-1> : tensor<1xi64>
  %ends   = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %axes   = onnx.Constant dense<1> : tensor<1xi64>
  %steps  = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
// CHECK-LABEL:  func.func @slice_dynamic_data_unchanged
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_2_]]) : (tensor<?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?xf32>
}

// -----
// Negative step: axis normalized (-1->0) and starts, but ends are left untouched.
func.func @slice_negative_step_axis_normalized_only(%arg0: tensor<3x2xi64>) -> tensor<3x2xi64> {
  %starts = onnx.Constant dense<-1> : tensor<1xi64>
  %ends   = onnx.Constant dense<-9223372036854775807> : tensor<1xi64>
  %axes   = onnx.Constant dense<0> : tensor<1xi64>
  %steps  = onnx.Constant dense<-1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) {onnx_node_name = "/Slice"} : (tensor<3x2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x2xi64>
  return %0 : tensor<3x2xi64>
// CHECK-LABEL:  func.func @slice_negative_step_axis_normalized_only
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xi64>) -> tensor<3x2xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-9223372036854775807> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_3_]], [[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) {onnx_node_name = "/Slice"} : (tensor<3x2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x2xi64>
// CHECK:           return [[VAR_4_]] : tensor<3x2xi64>
}

// -----
func.func @slice_neg_step_start_at_dim_clamps_to_dim_minus_one(%arg0: tensor<4xf32>) -> tensor<3xf32> {
  %starts = onnx.Constant dense<4> : tensor<1xi64>
  %ends   = onnx.Constant dense<0> : tensor<1xi64>
  %axes   = onnx.Constant dense<0> : tensor<1xi64>
  %steps  = onnx.Constant dense<-1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func.func @slice_neg_step_start_at_dim_clamps_to_dim_minus_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_2_]], [[VAR_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// CHECK:           return [[VAR_3_]] : tensor<3xf32>
}

// -----
// axes must materialise to [0, 1] (len=2, not len=4=rank).
// steps must materialise to [1, 1] (len=2, not len=4=rank).
// -----
func.func @slice_none_axes_steps_n_less_than_rank(%arg0: tensor<2x4x6x8xf32>) -> tensor<1x3x6x8xf32> {
  %starts = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %ends   = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %none, %none) : (tensor<2x4x6x8xf32>, tensor<2xi64>, tensor<2xi64>, none, none) -> tensor<1x3x6x8xf32>
  return %0 : tensor<1x3x6x8xf32>
// CHECK-LABEL:  func.func @slice_none_axes_steps_n_less_than_rank
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x6x8xf32>) -> tensor<1x3x6x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_0_]]) : (tensor<2x4x6x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x3x6x8xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x3x6x8xf32>
}
