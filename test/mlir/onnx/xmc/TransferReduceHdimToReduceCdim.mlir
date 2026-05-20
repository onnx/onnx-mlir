// RUN: onnx-mlir-opt --split-input-file --transfer-reduce-hdim-to-reduce-cdim %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Verifier for TransferReduceHdimToReduceCdimPass which contains four
// templated patterns that together shape ReduceSum/ReduceMean ops so the
// reduction axis is the last (channel/C-dim) axis -- the only form the AIE
// reduction kernel accepts.
//
//   1. ReduceHdimToCdimPattern         (rank-4 NCHW H-dim → NHWC C-dim)
//   2. ReduceWdimToCdimPattern         (rank-3 axis=1 with degenerate channel)
//   3. PadReduceTo4DPattern            (rank<4 → rank-4 + transpose-sandwich)
//   4. MoveReductionToLastAxisPattern  (rank-3 swap-reshape, walks through Cast)

//===----------------------------------------------------------------------===//
// Pattern 1: ReduceHdimToCdimPattern (rank-4 NCHW axis=[1] keepdims=true)
//===----------------------------------------------------------------------===//

// ReduceSum on rank-4 NCHW [N=1, C=8, H=4, W=4] axis=[1] keep_dims=true ->
// transpose [0,2,3,1] -> reduce on axis [3] -> transpose [0,3,1,2].

// CHECK-LABEL: @reduce_hdim_to_cdim_sum_rank4_channel
// CHECK-DAG:   %[[NEW_AXES:.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:       %[[T1:.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
// CHECK:       %[[R:.+]] = "onnx.ReduceSum"(%[[T1]], %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
// CHECK:       %[[T2:.+]] = "onnx.Transpose"(%[[R]]) {perm = [0, 3, 1, 2]}
// CHECK:       return %[[T2]]
func.func @reduce_hdim_to_cdim_sum_rank4_channel(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Same shape, ReduceMean variant -- the pattern is templated.

// CHECK-LABEL: @reduce_hdim_to_cdim_mean_rank4_channel
// CHECK:       "onnx.Transpose"({{.*}}) {perm = [0, 2, 3, 1]}
// CHECK:       "onnx.ReduceMean"
// CHECK-SAME:  keepdims = 1
// CHECK:       "onnx.Transpose"({{.*}}) {perm = [0, 3, 1, 2]}
func.func @reduce_hdim_to_cdim_mean_rank4_channel(%arg0: tensor<1x16x8x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x8x8x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x16x8x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x8x8x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x8x8x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Negative test: rank-4 reduce on axis [3] (already last) should NOT fire.

// CHECK-LABEL: @reduce_hdim_to_cdim_negative_axis_last
// CHECK-NOT:   onnx.Transpose
// CHECK:       "onnx.ReduceSum"
func.func @reduce_hdim_to_cdim_negative_axis_last(%arg0: tensor<1x1x768x150x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x768x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[3]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x768x150x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x768x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x768x1x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Negative test: rank-4 axis=[1] but keepdims=false should NOT fire
// (this pattern requires keep_dims=true; keepdims=false is left for
// PadReduceTo4DPattern when applicable, which only fires on rank<4).

// CHECK-LABEL: @reduce_hdim_to_cdim_negative_keepdims_false
// CHECK-NOT:   onnx.Transpose
// CHECK:       "onnx.ReduceSum"
// CHECK-SAME:  keepdims = 0
func.func @reduce_hdim_to_cdim_negative_keepdims_false(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Idempotency test: input is already the output of a transpose [0,2,3,1].
// Re-running the pattern would loop forever -- the guard must reject.

// CHECK-LABEL: @reduce_hdim_to_cdim_idempotency_guard
// CHECK:       %[[T:.+]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
// CHECK-NOT:   onnx.Transpose
// CHECK:       "onnx.ReduceSum"(%[[T]],
// CHECK:       return
func.func @reduce_hdim_to_cdim_idempotency_guard(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %t = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x4x8x!quant.uniform<i8:f32, 0.05:0>>
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%t, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x4x4x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

//===----------------------------------------------------------------------===//
// Pattern 2: ReduceWdimToCdimPattern (rank-3 axis=[1] keepdims=true, dim[2]=1)
//===----------------------------------------------------------------------===//

// Rank-3 [1, 150, 1] axis=[1] keep_dims=true -> reshape to [1, 1, 150] +
// reduce on axis [2].  Output [1, 1, 1] preserved exactly; no transpose.

// CHECK-LABEL: @reduce_wdim_to_cdim_sum_rank3_degenerate_channel
// CHECK-DAG:   %[[NEW_AXES:.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:   %[[NEW_SHAPE:.+]] = onnx.Constant dense<[1, 1, 150]> : tensor<3xi64>
// CHECK:       %[[R1:.+]] = "onnx.Reshape"(%arg0, %[[NEW_SHAPE]])
// CHECK-SAME:  -> tensor<1x1x150x!quant.uniform
// CHECK:       %[[R:.+]] = "onnx.ReduceSum"(%[[R1]], %[[NEW_AXES]]) {keepdims = 1 : si64
// CHECK-SAME:  -> tensor<1x1x1x!quant.uniform
// CHECK-NOT:   onnx.Transpose
// CHECK:       return %[[R]]
func.func @reduce_wdim_to_cdim_sum_rank3_degenerate_channel(%arg0: tensor<1x150x1x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x150x1x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x1x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Negative test: rank-3 axis=[1] keepdims=true but channel dim != 1 -- the
// swap would not be a free view, pattern must refuse.

// CHECK-LABEL: @reduce_wdim_to_cdim_negative_channel_not_one
// CHECK-NOT:   onnx.Reshape
// CHECK-NOT:   onnx.Transpose
// CHECK:       "onnx.ReduceSum"
func.func @reduce_wdim_to_cdim_negative_channel_not_one(%arg0: tensor<1x150x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x150x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Negative test: rank-3 axis=[1] keepdims=true with shape[1] = 1 (reduction
// is a no-op already) -- pattern must refuse.

// CHECK-LABEL: @reduce_wdim_to_cdim_negative_reduce_dim_one
// CHECK-NOT:   onnx.Reshape
// CHECK:       "onnx.ReduceSum"
func.func @reduce_wdim_to_cdim_negative_reduce_dim_one(%arg0: tensor<1x1x1x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x1x1x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x1x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

//===----------------------------------------------------------------------===//
// Pattern 3: PadReduceTo4DPattern (rank<4 padded then sandwiched)
//===----------------------------------------------------------------------===//

// Rank-3 [150, 1, 768] axis=[0] keepdims=false -> pad to rank-4
// [1, 150, 1, 768], shift axis to [1], keepdims forced to true, then
// ReduceHdimToCdimPattern fires producing the transpose-sandwich, with a
// trailing reshape restoring the original output shape [1, 768].

// CHECK-LABEL: @pad_reduce_to_4d_rank3_axis0
// CHECK-DAG:   %[[OUT_SHAPE:.+]] = onnx.Constant dense<[1, 768]> : tensor<2xi64>
// CHECK-DAG:   %[[NEW_AXES:.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:   %[[PAD_SHAPE:.+]] = onnx.Constant dense<[1, 150, 1, 768]> : tensor<4xi64>
// CHECK:       %[[PAD:.+]] = "onnx.Reshape"(%arg0, %[[PAD_SHAPE]])
// CHECK-SAME:  -> tensor<1x150x1x768x
// CHECK:       %[[T1:.+]] = "onnx.Transpose"(%[[PAD]]) {perm = [0, 2, 3, 1]}
// CHECK:       %[[R:.+]] = "onnx.ReduceSum"(%[[T1]], %[[NEW_AXES]]) {keepdims = 1 : si64
// CHECK:       %[[T2:.+]] = "onnx.Transpose"(%[[R]]) {perm = [0, 3, 1, 2]}
// CHECK:       %[[OUT:.+]] = "onnx.Reshape"(%[[T2]], %[[OUT_SHAPE]])
// CHECK-SAME:  -> tensor<1x768x
// CHECK:       return %[[OUT]]
func.func @pad_reduce_to_4d_rank3_axis0(%arg0: tensor<150x1x768x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x768x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[0]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<150x1x768x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x768x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x768x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Negative test: rank-3 axis=[1] (would shift to axis=[2] after padding,
// which ReduceHdimToCdimPattern does not handle).  PadReduceTo4DPattern
// only fires when the shifted axis lands on position 1.  Pattern 4
// (MoveReductionToLastAxisPattern) also refuses because shape[2] != 1.

// CHECK-LABEL: @pad_reduce_to_4d_negative_axis_not_landing_on_1
// CHECK-NOT:   onnx.Transpose
// CHECK-NOT:   onnx.Reshape
// CHECK:       "onnx.ReduceSum"
func.func @pad_reduce_to_4d_negative_axis_not_landing_on_1(%arg0: tensor<2x150x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<2x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x150x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<2x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<2x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

//===----------------------------------------------------------------------===//
// Pattern 4: MoveReductionToLastAxisPattern (rank-3 swap, walks through Cast)
//===----------------------------------------------------------------------===//

// Rank-3 [1, 150, 1] axis=[1] keepdims=false (no Cast wrapping) -> swap
// reshape [1, 1, 150] + reduce on axis [2] keepdims=false.  Output [1, 1]
// preserved exactly.

// CHECK-LABEL: @move_reduction_to_last_axis_rank3_keepdims_false_no_cast
// CHECK-DAG:   %[[NEW_AXES:.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:   %[[NEW_SHAPE:.+]] = onnx.Constant dense<[1, 1, 150]> : tensor<3xi64>
// CHECK:       %[[R1:.+]] = "onnx.Reshape"(%arg0, %[[NEW_SHAPE]])
// CHECK-SAME:  -> tensor<1x1x150x!quant.uniform
// CHECK:       %[[R:.+]] = "onnx.ReduceSum"(%[[R1]], %[[NEW_AXES]]) {keepdims = 0 : si64
// CHECK-SAME:  -> tensor<1x1x!quant.uniform
// CHECK-NOT:   onnx.Transpose
// CHECK:       return %[[R]]
func.func @move_reduction_to_last_axis_rank3_keepdims_false_no_cast(%arg0: tensor<1x150x1x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x150x1x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Cast-walk-through case: Cast(i32->i64) -> ReduceSum(i64) on rank-3
// [1,150,1] axis=[1] keepdims=false.  The reshape lands on the i32 input
// and the Cast is re-emitted on the swapped shape.

// CHECK-LABEL: @move_reduction_to_last_axis_rank3_walks_through_cast
// CHECK-DAG:   %[[NEW_AXES:.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:   %[[NEW_SHAPE:.+]] = onnx.Constant dense<[1, 1, 150]> : tensor<3xi64>
// CHECK:       %[[R1:.+]] = "onnx.Reshape"(%arg0, %[[NEW_SHAPE]])
// CHECK-SAME:  (tensor<1x150x1xi32>
// CHECK-SAME:  -> tensor<1x1x150xi32>
// CHECK:       %[[C:.+]] = "onnx.Cast"(%[[R1]]) {saturate = 1 : si64, to = i64}
// CHECK-SAME:  -> tensor<1x1x150xi64>
// CHECK:       %[[R:.+]] = "onnx.ReduceSum"(%[[C]], %[[NEW_AXES]]) {keepdims = 0 : si64
// CHECK-SAME:  -> tensor<1x1xi64>
// CHECK:       return %[[R]]
func.func @move_reduction_to_last_axis_rank3_walks_through_cast(%arg0: tensor<1x150x1xi32>) -> tensor<1x1xi64> {
    %c = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i64} : (tensor<1x150x1xi32>) -> tensor<1x150x1xi64>
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%c, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x150x1xi64>, tensor<1xi64>) -> tensor<1x1xi64>
    return %1 : tensor<1x1xi64>
}

// -----

// Negative test: rank-3 axis=[1] keepdims=false but channel dim != 1.

// CHECK-LABEL: @move_reduction_to_last_axis_negative_channel_not_one
// CHECK-NOT:   onnx.Reshape
// CHECK:       "onnx.ReduceSum"
func.func @move_reduction_to_last_axis_negative_channel_not_one(%arg0: tensor<1x150x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x150x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Negative test: rank-3 axis=[2] (already on the last axis).

// CHECK-LABEL: @move_reduction_to_last_axis_negative_axis_already_last
// CHECK-NOT:   onnx.Reshape
// CHECK:       "onnx.ReduceSum"
func.func @move_reduction_to_last_axis_negative_axis_already_last(%arg0: tensor<1x4x150x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x4x150x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x4x!quant.uniform<i8:f32, 0.05:0>>
}

