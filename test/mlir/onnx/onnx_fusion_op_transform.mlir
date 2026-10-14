// RUN: onnx-mlir-opt --fusion-op-transform %s -split-input-file | FileCheck %s

// Tests for the "simd-split-op-gather" ONNXFusedOp pattern, detected by the
// (CPU-general, non-accelerator-specific) --fusion-op-transform pass. The
// pass wraps a chain of:
//   ONNXSliceOp (low half) -- [op] --\
//                                      onnx.Concat
//   ONNXSliceOp (high half) -- [op] -/
// into an onnx.Fused region with kind = "simd-split-op-gather", when the two
// slices are dense, contiguous, and together cover the whole (innermost)
// split axis, each optional per-half op is a single elementwise op with at
// most one extra same-shape operand not depending on the other half, and
// Concat has exactly those two (possibly transformed) values as its only
// two operands. See src/Dialect/ONNX/Transforms/ONNXFusionOpHelper.hpp.
//
// Since there is no dedicated Krnl lowering yet, every fused op below would
// currently be inlined back to its original ops by FusedOpInlineFallback at
// --convert-onnx-to-krnl time -- these tests only cover detection.

// -----

// Positive: the RoPE rotate_half idiom, exactly as seen in Granite-4 --
// including the INT64_MAX "slice to the end of the axis" sentinel on the
// high half's `end` operand. Op (Neg) is on the high half only; Concat
// reorders the halves (high first), so outputOffsetForSplitHigh=0 even
// though the high half itself starts at splitPoint=64 on the input side.

func.func @rotate_half_int64_max_sentinel(%arg0: tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c64 = onnx.Constant dense<64> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c64, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %1 = "onnx.Slice"(%arg0, %c64, %cmax, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %2 = "onnx.Neg"(%1) : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 3 : si64} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
  return %3 : tensor<1x4x8x128xf32>

// CHECK-LABEL:  func.func @rotate_half_int64_max_sentinel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4x8x128xf32>)
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]]) <{kind = "simd-split-op-gather"}> ({
// CHECK:           ^bb0([[PARAM_1_:%.+]]: tensor<1x4x8x128xf32>):
// CHECK-DAG:           "onnx.Slice"([[PARAM_1_]], {{.*}}) : {{.*}} -> tensor<1x4x8x64xf32>
// CHECK-DAG:           "onnx.Slice"([[PARAM_1_]], {{.*}}) : {{.*}} -> tensor<1x4x8x64xf32>
// CHECK:               "onnx.Neg"
// CHECK:               "onnx.Concat"
// CHECK:               onnx.Yield
// CHECK:           }) {axis = 3 : i64, hasOpForSplitHigh = true, hasOpForSplitLow = false, {{.*}}outputOffsetForSplitHigh = 0 : i64, outputOffsetForSplitLow = 64 : i64, splitPoint = 64 : i64}
// CHECK:           return [[VAR_0_]]
}

// -----

// Positive: op on the LOW half only, high half copied as-is, Concat keeps
// the input order (low first) -- exercises hasOpForSplitLow with an
// output-offset-matches-input-offset layout (unlike rotate_half's swap).

func.func @op_on_low_half_only(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Relu"(%0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Concat"(%2, %1) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  return %3 : tensor<2x8xf32>

// CHECK-LABEL:  func.func @op_on_low_half_only
// CHECK:           "onnx.Fused"({{.*}}) <{kind = "simd-split-op-gather"}> ({
// CHECK:               "onnx.Relu"
// CHECK:               onnx.Yield
// CHECK:           }) {axis = 1 : i64, hasOpForSplitHigh = false, hasOpForSplitLow = true, {{.*}}outputOffsetForSplitHigh = 4 : i64, outputOffsetForSplitLow = 0 : i64, splitPoint = 4 : i64}
}

// -----

// Positive: both halves have an op -- exercises hasOpForSplitLow AND
// hasOpForSplitHigh simultaneously (5 chain ops instead of 3 or 4).

func.func @ops_on_both_halves(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Neg"(%0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Relu"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %4 = "onnx.Concat"(%2, %3) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  return %4 : tensor<2x8xf32>

// CHECK-LABEL:  func.func @ops_on_both_halves
// CHECK:           "onnx.Fused"({{.*}}) <{kind = "simd-split-op-gather"}> ({
// CHECK:               "onnx.Neg"
// CHECK:               "onnx.Relu"
// CHECK:               "onnx.Concat"
// CHECK:               onnx.Yield
// CHECK:           }) {axis = 1 : i64, hasOpForSplitHigh = true, hasOpForSplitLow = true, {{.*}}outputOffsetForSplitHigh = 4 : i64, outputOffsetForSplitLow = 0 : i64, splitPoint = 4 : i64}
}

// -----

// Positive: binary per-half op with an external, same-shape operand --
// exercises the FusedOp getting a second (non-split-source) input.

func.func @scaled_half_with_external_operand(%arg0: tensor<2x8xf32>, %scale: tensor<2x4xf32>) -> tensor<2x8xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Mul"(%1, %scale) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Concat"(%0, %2) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  return %3 : tensor<2x8xf32>

// CHECK-LABEL:  func.func @scaled_half_with_external_operand
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4xf32>)
// CHECK:           "onnx.Fused"([[PARAM_0_]], [[PARAM_1_]]) <{kind = "simd-split-op-gather"}> ({
// CHECK:           ^bb0([[PARAM_2_:%.+]]: tensor<2x8xf32>, [[PARAM_3_:%.+]]: tensor<2x4xf32>):
// CHECK:               "onnx.Mul"({{.*}}, [[PARAM_3_]])
// CHECK:               onnx.Yield
// CHECK:           }) {axis = 1 : i64, hasOpForSplitHigh = true, hasOpForSplitLow = false, {{.*}}splitPoint = 4 : i64}
}

// -----

// Same, switched the two slices. It normalizes the order.

func.func @scaled_half_with_external_operand_switched(%arg0: tensor<2x8xf32>, %scale: tensor<2x4xf32>) -> tensor<2x8xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Mul"(%1, %scale) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Concat"(%0, %2) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  return %3 : tensor<2x8xf32>

// CHECK-LABEL:  func.func @scaled_half_with_external_operand_switched
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4xf32>) -> tensor<2x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]], [[PARAM_1_]]) <{kind = "simd-split-op-gather"}> ({
// CHECK:           ^bb0([[arg2_:%.+]]: tensor<2x8xf32>, [[arg3_:%.+]]: tensor<2x4xf32>):
// CHECK-DAG:         [[VAR_1_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
// CHECK-DAG:         [[VAR_2_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = "onnx.Slice"([[arg2_]], [[VAR_2_]], [[VAR_3_]], [[VAR_4_]], [[VAR_4_]]) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = "onnx.Slice"([[arg2_]], [[VAR_3_]], [[VAR_1_]], [[VAR_4_]], [[VAR_4_]]) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
// CHECK:             [[VAR_7_:%.+]] = "onnx.Mul"([[VAR_6_]], [[arg3_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:             [[VAR_8_:%.+]] = "onnx.Concat"([[VAR_5_]], [[VAR_7_]]) <{axis = 1 : si64}> : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
// CHECK:             onnx.Yield [[VAR_8_]] : tensor<2x8xf32>
// CHECK:           }) {axis = 1 : i64, hasOpForSplitHigh = true, hasOpForSplitLow = false, onnx_node_name = "onnx.Slice-onnx.Slice-onnx.Mul-onnx.Concat", outputOffsetForSplitHigh = 4 : i64, outputOffsetForSplitLow = 0 : i64, splitPoint = 4 : i64} : (tensor<2x8xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
// CHECK:           return [[VAR_0_]] : tensor<2x8xf32>
// CHECK:         }
}

// -----

// Negative: a 2-element gap between the two slices ([0,3) and [4,8)).

func.func @no_fuse_gap_between_slices(%arg0: tensor<2x8xf32>) -> tensor<2x7xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c3, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Neg"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<2x7xf32>
  return %3 : tensor<2x7xf32>

// CHECK-LABEL:  func.func @no_fuse_gap_between_slices
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Neg"
// CHECK:           "onnx.Concat"
// CHECK:           return
}

// -----

// Negative: the two slices overlap ([0,5) and [4,8)).

func.func @no_fuse_overlapping_slices(%arg0: tensor<2x8xf32>) -> tensor<2x9xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c5 = onnx.Constant dense<5> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c5, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x5xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Neg"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x5xf32>) -> tensor<2x9xf32>
  return %3 : tensor<2x9xf32>

// CHECK-LABEL:  func.func @no_fuse_overlapping_slices
// CHECK-NOT:       "onnx.Fused"
// CHECK:           return
}

// -----

// Negative: the per-half op (Softmax) is not elementwise, so it's not in
// the v1 allow-list.

func.func @no_fuse_disallowed_op(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Softmax"(%1) {axis = -1 : si64} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  return %3 : tensor<2x8xf32>

// CHECK-LABEL:  func.func @no_fuse_disallowed_op
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Softmax"
// CHECK:           return
}

// -----

// Negative: Concat has a third operand -- not a 2-way split/gather.

func.func @no_fuse_three_way_concat(%arg0: tensor<2x8xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x10xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Neg"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Concat"(%2, %0, %arg1) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x2xf32>) -> tensor<2x10xf32>
  return %3 : tensor<2x10xf32>

// CHECK-LABEL:  func.func @no_fuse_three_way_concat
// CHECK-NOT:       "onnx.Fused"
// CHECK:           return
}

// -----

// Negative: the split axis (0) is not the innermost dim -- not supported
// in v1.

func.func @no_fuse_non_innermost_axis(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c0axis = onnx.Constant dense<0> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c0axis, %c1) : (tensor<8x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c0axis, %c1) : (tensor<8x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4x4xf32>
  %2 = "onnx.Neg"(%1) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 0 : si64} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
  return %3 : tensor<8x4xf32>

// CHECK-LABEL:  func.func @no_fuse_non_innermost_axis
// CHECK-NOT:       "onnx.Fused"
// CHECK:           return
}

// -----

// Negative: the per-half op's external operand is literally the OTHER
// half's slice result -- a direct cross-half dependency.

func.func @no_fuse_cross_half_dependency(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %2 = "onnx.Mul"(%1, %0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 1 : si64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
  return %3 : tensor<2x8xf32>

// CHECK-LABEL:  func.func @no_fuse_cross_half_dependency
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Mul"
// CHECK:           return
}

// -----

// Positive: dynamic batch and sequence-length dims (axis 0 and 2), exactly
// the shape from the Granite-4 IR this pattern is modeled on -- only the
// split axis (3, size 128) needs to be static.

func.func @dynamic_non_split_dims(%arg0: tensor<?x4x?x128xf32>) -> tensor<?x4x?x128xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c64 = onnx.Constant dense<64> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c64, %c3, %c1) : (tensor<?x4x?x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x?x64xf32>
  %1 = "onnx.Slice"(%arg0, %c64, %cmax, %c3, %c1) : (tensor<?x4x?x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x?x64xf32>
  %2 = "onnx.Neg"(%1) : (tensor<?x4x?x64xf32>) -> tensor<?x4x?x64xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 3 : si64} : (tensor<?x4x?x64xf32>, tensor<?x4x?x64xf32>) -> tensor<?x4x?x128xf32>
  return %3 : tensor<?x4x?x128xf32>

// CHECK-LABEL:   func.func @dynamic_non_split_dims(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x4x?x128xf32>) -> tensor<?x4x?x128xf32> {
// CHECK:           %[[VAL_0:.*]] = "onnx.Fused"(%[[ARG0]]) <{kind = "simd-split-op-gather"}> ({
// CHECK:           ^bb0(%[[VAL_1:.*]]: tensor<?x4x?x128xf32>):
// CHECK:             %[[VAL_2:.*]] = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
// CHECK:             %[[VAL_3:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:             %[[VAL_4:.*]] = onnx.Constant dense<64> : tensor<1xi64>
// CHECK:             %[[VAL_5:.*]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:             %[[VAL_6:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:             %[[VAL_7:.*]] = "onnx.Slice"(%[[VAL_1]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]]) : (tensor<?x4x?x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x?x64xf32>
// CHECK:             %[[VAL_8:.*]] = "onnx.Slice"(%[[VAL_1]], %[[VAL_4]], %[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) : (tensor<?x4x?x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x?x64xf32>
// CHECK:             %[[VAL_9:.*]] = "onnx.Neg"(%[[VAL_8]]) : (tensor<?x4x?x64xf32>) -> tensor<?x4x?x64xf32>
// CHECK:             %[[VAL_10:.*]] = "onnx.Concat"(%[[VAL_9]], %[[VAL_7]]) <{axis = 3 : si64}> : (tensor<?x4x?x64xf32>, tensor<?x4x?x64xf32>) -> tensor<?x4x?x128xf32>
// CHECK:             onnx.Yield %[[VAL_10]] : tensor<?x4x?x128xf32>
// CHECK:           }) {axis = 3 : i64, hasOpForSplitHigh = true, hasOpForSplitLow = false, onnx_node_name = "onnx.Slice-onnx.Slice-onnx.Neg-onnx.Concat", outputOffsetForSplitHigh = 0 : i64, outputOffsetForSplitLow = 64 : i64, splitPoint = 64 : i64} : (tensor<?x4x?x128xf32>) -> tensor<?x4x?x128xf32>
// CHECK:           return %[[VAL_0]] : tensor<?x4x?x128xf32>
// CHECK:         }
}

// -----

// Positive: binary per-half op whose external operand has a dynamic dim
// (axis 0) -- proven, via onnx.dim_params ("M" on both %arg0 and %scale), to
// be the exact same runtime dimension as the slice's own dynamic dim. Same
// idea as @scaled_half_with_external_operand, but for a dim that DimAnalysis
// must actively prove equal rather than one that's trivially static.

func.func @scaled_half_with_external_dynamic_matching(%arg0: tensor<?x8xf32> {onnx.dim_params = "0:M"}, %scale: tensor<?x4xf32> {onnx.dim_params = "0:M"}) -> tensor<?x8xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<?x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<?x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4xf32>
  %2 = "onnx.Mul"(%1, %scale) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %3 = "onnx.Concat"(%0, %2) {axis = 1 : si64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x8xf32>
  return %3 : tensor<?x8xf32>

// CHECK-LABEL:  func.func @scaled_half_with_external_dynamic_matching
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x8xf32> {onnx.dim_params = "0:M"}, [[PARAM_1_:%.+]]: tensor<?x4xf32> {onnx.dim_params = "0:M"})
// CHECK:           "onnx.Fused"([[PARAM_0_]], [[PARAM_1_]]) <{kind = "simd-split-op-gather"}> ({
// CHECK:           ^bb0([[PARAM_2_:%.+]]: tensor<?x8xf32>, [[PARAM_3_:%.+]]: tensor<?x4xf32>):
// CHECK:               "onnx.Mul"({{.*}}, [[PARAM_3_]])
// CHECK:               onnx.Yield
// CHECK:           }) {axis = 1 : i64, hasOpForSplitHigh = true, hasOpForSplitLow = false, {{.*}}splitPoint = 4 : i64}
}

// -----

// Negative: binary per-half op whose external operand has a dynamic dim
// (axis 0) that is NOT proven the same as the slice's own dynamic dim --
// %scale's batch dim is a plain, unrelated dynamic dim (no dim_params link
// to %arg0's). Both dims print as "?" in the type, but that textual sameness
// does not mean they're the same runtime value: %scale's dim 0 could well be
// 1 at runtime, in which case an un-fused Mul legitimately broadcasts it
// against the slice's batch dim -- valid ONNX semantics, but not something
// this v1 (no-broadcast) fusion can support. So it must decline to fuse
// rather than assume the two "?"s denote equal dims.

func.func @no_fuse_external_dynamic_not_proven_same(%arg0: tensor<?x8xf32>, %scale: tensor<?x4xf32>) -> tensor<?x8xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c1axis = onnx.Constant dense<1> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c1axis, %c1) : (tensor<?x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c1axis, %c1) : (tensor<?x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4xf32>
  %2 = "onnx.Mul"(%1, %scale) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %3 = "onnx.Concat"(%0, %2) {axis = 1 : si64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x8xf32>
  return %3 : tensor<?x8xf32>

// CHECK-LABEL:  func.func @no_fuse_external_dynamic_not_proven_same
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Mul"
// CHECK:           return
}

// -----

// Negative: the SPLIT axis itself (3) is dynamic -- required static (it's
// what lets the "slice to the end" sentinel resolve to a literal, and what
// Part 2's lowering needs for a compile-time SIMD length), so this must not
// fuse even though everything else about the pattern matches.

func.func @no_fuse_dynamic_split_axis(%arg0: tensor<?x4x8x?xf32>) -> tensor<?x4x8x?xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c4 = onnx.Constant dense<4> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c4, %c3, %c1) : (tensor<?x4x8x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x8x?xf32>
  %1 = "onnx.Slice"(%arg0, %c4, %cmax, %c3, %c1) : (tensor<?x4x8x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x8x?xf32>
  %2 = "onnx.Neg"(%1) : (tensor<?x4x8x?xf32>) -> tensor<?x4x8x?xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 3 : si64} : (tensor<?x4x8x?xf32>, tensor<?x4x8x?xf32>) -> tensor<?x4x8x?xf32>
  return %3 : tensor<?x4x8x?xf32>

// CHECK-LABEL:   func.func @no_fuse_dynamic_split_axis(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x4x8x?xf32>) -> tensor<?x4x8x?xf32> {
// CHECK-NOT:       "onnx.Fused"
// CHECK:           %[[VAL_0:.*]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           %[[VAL_1:.*]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK:           %[[VAL_2:.*]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           %[[VAL_3:.*]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           %[[VAL_4:.*]] = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
// CHECK:           %[[VAL_5:.*]] = "onnx.Slice"(%[[ARG0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) : (tensor<?x4x8x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x8x?xf32>
// CHECK:           %[[VAL_6:.*]] = "onnx.Slice"(%[[ARG0]], %[[VAL_1]], %[[VAL_4]], %[[VAL_2]], %[[VAL_3]]) : (tensor<?x4x8x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x8x?xf32>
// CHECK:           %[[VAL_7:.*]] = "onnx.Neg"(%[[VAL_6]]) : (tensor<?x4x8x?xf32>) -> tensor<?x4x8x?xf32>
// CHECK:           %[[VAL_8:.*]] = "onnx.Concat"(%[[VAL_7]], %[[VAL_5]]) <{axis = 3 : si64}> : (tensor<?x4x8x?xf32>, tensor<?x4x8x?xf32>) -> tensor<?x4x8x?xf32>
// CHECK:           return %[[VAL_8]] : tensor<?x4x8x?xf32>
// CHECK:         }
}
