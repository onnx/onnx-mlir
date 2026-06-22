// RUN: onnx-mlir-opt --split-input-file --replace-qdq-reduction %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Tests for ReplaceQDQReductionPass: canonicalises Q/DQ-bracketed
// Reduce(Sum/Mean/Max/Min/ReduceMaxV13) to rank-4 + keep_dims=true.

// Branch 1: rank<4 -- prepend size-1 dim, bump axis (PSO3-style).
// CHECK-LABEL: @rank3_axis0_keepdims0_pso3
// CHECK-DAG:   %[[PRESHAPE_C:.+]] = onnx.Constant dense<[1, 150, 1, 768]> : tensor<4xi64>
// CHECK-DAG:   %[[NEW_AXES:.+]]   = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[1, 768]> : tensor<2xi64>
// CHECK:       %[[PRE:.+]]  = "onnx.Reshape"(%arg0, %[[PRESHAPE_C]]) {{.*}} : (tensor<150x1x768x{{.*}}>, tensor<4xi64>) -> tensor<1x150x1x768x{{.*}}>
// CHECK:       %[[RED:.+]]  = "onnx.ReduceSum"(%[[PRE]], %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x150x1x768x{{.*}}>, tensor<1xi64>) -> tensor<1x1x1x768x{{.*}}>
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]]) {{.*}} : (tensor<1x1x1x768x{{.*}}>, tensor<2xi64>) -> tensor<1x768x{{.*}}>
// CHECK:       return %[[POST]]
func.func @rank3_axis0_keepdims0_pso3(%arg0: tensor<150x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>) -> tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>> {
  %0 = onnx.Constant dense<[0]> : tensor<1xi64>
  %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<150x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>, tensor<1xi64>) -> tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>
  return %1 : tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>
}

// -----

// rank-3 ReduceMean (4x8x16) axes=[1] keepdims=1 -> (4x1x16).  Becomes:
// CHECK-LABEL: @rank3_mean_axis1_keepdims1_template
// CHECK-DAG:   %[[PRESHAPE_C:.+]] = onnx.Constant dense<[1, 4, 8, 16]> : tensor<4xi64>
// CHECK-DAG:   %[[NEW_AXES:.+]]   = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[4, 1, 16]> : tensor<3xi64>
// CHECK:       %[[PRE:.+]]  = "onnx.Reshape"(%arg0, %[[PRESHAPE_C]])
// CHECK:       %[[RED:.+]]  = "onnx.ReduceMean"(%[[PRE]], %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]])
// CHECK:       return %[[POST]]
func.func @rank3_mean_axis1_keepdims1_template(%arg0: tensor<4x8x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<4x8x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Branch 2: rank==4 + keep_dims=true -- early-exit (already canonical).

// CHECK-LABEL: @rank4_keepdims1_already_canonical
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"
// CHECK-SAME:  keepdims = 1
// CHECK-SAME:  (tensor<1x8x4x4x{{.*}}>, tensor<1xi64>) -> tensor<1x1x4x4x{{.*}}>
func.func @rank4_keepdims1_already_canonical(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Branch 3: rank==4 + keep_dims=false -- re-emit with keepdims=true + trailing reshape.

// CHECK-LABEL: @rank4_keepdims0_reemit
// CHECK-DAG:   %[[NEW_AXES:.+]]    = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[1, 4, 4]> : tensor<3xi64>
// CHECK:       %[[RED:.+]]  = "onnx.ReduceSum"(%arg0, %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x{{.*}}>, tensor<1xi64>) -> tensor<1x1x4x4x{{.*}}>
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]])
// CHECK:       return %[[POST]]
func.func @rank4_keepdims0_reemit(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Branch 4: rank>4 with leading-1 + last_axis in {2,3} -- drop leading 1, decrement axes.

// rank-5 ReduceMax (1x2x4x8x16) axes=[2] keepdims=1 -> (1x2x1x8x16).  Becomes:
// CHECK-LABEL: @rank5_leading1_lastaxis_in_2_3
// CHECK-DAG:   %[[PRESHAPE_C:.+]]  = onnx.Constant dense<[2, 4, 8, 16]> : tensor<4xi64>
// CHECK-DAG:   %[[NEW_AXES:.+]]    = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[1, 2, 1, 8, 16]> : tensor<5xi64>
// CHECK:       %[[PRE:.+]]  = "onnx.Reshape"(%arg0, %[[PRESHAPE_C]])
// CHECK:       %[[RED:.+]]  = "onnx.ReduceMax"(%[[PRE]], %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]])
// CHECK:       return %[[POST]]
func.func @rank5_leading1_lastaxis_in_2_3(%arg0: tensor<1x2x4x8x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x2x1x8x16x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[2]> : tensor<1xi64>
  %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x4x8x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x2x1x8x16x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<1x2x1x8x16x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Branch 5: rank>4 NOT leading-1 -- collapse middle dims (single-axis on last).

// rank-5 ReduceMin (2x3x4x5x6) axes=[4] (last) keepdims=1 -> (2x3x4x5x1).  Becomes:
// CHECK-LABEL: @rank5_collapse_middle_dims
// CHECK-DAG:   %[[PRESHAPE_C:.+]]  = onnx.Constant dense<[2, 3, 20, 6]> : tensor<4xi64>
// CHECK-DAG:   %[[NEW_AXES:.+]]    = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[2, 3, 4, 5, 1]> : tensor<5xi64>
// CHECK:       %[[PRE:.+]]  = "onnx.Reshape"(%arg0, %[[PRESHAPE_C]])
// CHECK:       %[[RED:.+]]  = "onnx.ReduceMin"(%[[PRE]], %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]])
// CHECK:       return %[[POST]]
func.func @rank5_collapse_middle_dims(%arg0: tensor<2x3x4x5x6x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<2x3x4x5x1x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[4]> : tensor<1xi64>
  %1 = "onnx.ReduceMin"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4x5x6x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<2x3x4x5x1x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<2x3x4x5x1x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Negative tests

// Not in Q/DQ chain (f32 input + f32 output, no quant types) -- skip.
// CHECK-LABEL: @negative_no_qdq_chain
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"
// CHECK-SAME:  keepdims = 0
func.func @negative_no_qdq_chain(%arg0: tensor<150x1x768xf32>) -> tensor<1x768xf32> {
  %0 = onnx.Constant dense<[0]> : tensor<1xi64>
  %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<150x1x768xf32>, tensor<1xi64>) -> tensor<1x768xf32>
  return %1 : tensor<1x768xf32>
}

// -----

// Multi-fanout reduction (used twice) -- the get_template gate (single
// fanout) blocks the rewrite to preserve xcompiler.git's downstream
// `replace_reduce` template match.
// CHECK-LABEL: @negative_multi_fanout
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"
func.func @negative_multi_fanout(%arg0: tensor<150x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>) -> (tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>, tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>) {
  %0 = onnx.Constant dense<[0]> : tensor<1xi64>
  %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<150x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>, tensor<1xi64>) -> tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>
  return %1, %1 : tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>, tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>
}

// -----

// Dynamic shape -- skip (the algorithm requires static shapes for its
// reshape sizes).
// CHECK-LABEL: @negative_dynamic_shape
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"
func.func @negative_dynamic_shape(%arg0: tensor<?x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>) -> tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>> {
  %0 = onnx.Constant dense<[0]> : tensor<1xi64>
  %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>, tensor<1xi64>) -> tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>
  return %1 : tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>
}

// -----

// Non-constant axes -- skip (the algorithm requires constant axes to plan
// the reshape).
// CHECK-LABEL: @negative_non_constant_axes
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"
func.func @negative_non_constant_axes(%arg0: tensor<150x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>, %axes: tensor<1xi64>) -> tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>> {
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<150x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>, tensor<1xi64>) -> tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>
  return %0 : tensor<1x768x!quant.uniform<u16:f32, 7.842E-3:40316>>
}

// -----

// Coverage: Sum / Mean / Max / Min / ReduceMaxV13 share the algorithm; this
// section adds rank<4 Max, Min, and ReduceMaxV13 to confirm registration.

// CHECK-LABEL: @rank3_max_axis1_keepdims1
// CHECK:       "onnx.Reshape"
// CHECK:       "onnx.ReduceMax"
// CHECK-SAME:  keepdims = 1
// CHECK:       "onnx.Reshape"
func.func @rank3_max_axis1_keepdims1(%arg0: tensor<4x8x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<4x8x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// ReduceMaxV13: `axes` is an attribute (not an SSA operand); same reshape
// plan as @rank3_max_axis1_keepdims1.
// CHECK-LABEL: @rank3_reducemaxv13_axis1_keepdims1
// CHECK-DAG:   %[[PRESHAPE_C:.+]] = onnx.Constant dense<[1, 4, 8, 16]> : tensor<4xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[4, 1, 16]> : tensor<3xi64>
// CHECK:       %[[PRE:.+]]  = "onnx.Reshape"(%arg0, %[[PRESHAPE_C]])
// CHECK:       %[[RED:.+]]  = "onnx.ReduceMaxV13"(%[[PRE]]){{.*}}axes = [2], keepdims = 1 : si64
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]])
// CHECK-SAME:  ResultNames = ["rank3_reducemaxv13_out"]
// CHECK:       return %[[POST]]
func.func @rank3_reducemaxv13_axis1_keepdims1(%arg0: tensor<4x8x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = "onnx.ReduceMaxV13"(%arg0) {axes = [1], keepdims = 1 : si64, ResultNames = ["rank3_reducemaxv13_out"]} : (tensor<4x8x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>>
  return %0 : tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// CHECK-LABEL: @rank3_min_axis1_keepdims1
// CHECK:       "onnx.Reshape"
// CHECK:       "onnx.ReduceMin"
// CHECK-SAME:  keepdims = 1
// CHECK:       "onnx.Reshape"
func.func @rank3_min_axis1_keepdims1(%arg0: tensor<4x8x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[1]> : tensor<1xi64>
  %1 = "onnx.ReduceMin"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<4x8x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<4x1x16x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Branch 3: rank==4 + keep_dims=false, axes=[-1].
// Same expected IR as @rank4_keepdims0_reemit but with axes=[3] (rank-1).
// CHECK-LABEL: @rank4_keepdims0_negative_axis
// CHECK-DAG:   %[[NEW_AXES:.+]]    = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[1, 8, 4]> : tensor<3xi64>
// CHECK:       %[[RED:.+]]  = "onnx.ReduceSum"(%arg0, %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x{{.*}}>, tensor<1xi64>) -> tensor<1x8x4x1x{{.*}}>
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]])
// CHECK:       return %[[POST]]
func.func @rank4_keepdims0_negative_axis(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x4x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[-1]> : tensor<1xi64>
  %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x8x4x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<1x8x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Branch 3: rank==4 + keep_dims=false, axes=[-3].  Equivalent to axes=[1];
// must produce IR identical to @rank4_keepdims0_reemit.
// CHECK-LABEL: @rank4_keepdims0_negative_axis_mid
// CHECK-DAG:   %[[NEW_AXES:.+]]    = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[1, 4, 4]> : tensor<3xi64>
// CHECK:       %[[RED:.+]]  = "onnx.ReduceSum"(%arg0, %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x{{.*}}>, tensor<1xi64>) -> tensor<1x1x4x4x{{.*}}>
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]])
// CHECK:       return %[[POST]]
func.func @rank4_keepdims0_negative_axis_mid(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[-3]> : tensor<1xi64>
  %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Branch 1: rank<4, axes=[-1] keepdims=0.  -1 normalises to 2 then the
// rank-3->rank-4 prepend shifts it to 3.
// CHECK-LABEL: @rank3_negative_axis_keepdims0
// CHECK-DAG:   %[[PRESHAPE_C:.+]]  = onnx.Constant dense<[1, 150, 1, 768]> : tensor<4xi64>
// CHECK-DAG:   %[[NEW_AXES:.+]]    = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[150, 1]> : tensor<2xi64>
// CHECK:       %[[PRE:.+]]  = "onnx.Reshape"(%arg0, %[[PRESHAPE_C]]) {{.*}} : (tensor<150x1x768x{{.*}}>, tensor<4xi64>) -> tensor<1x150x1x768x{{.*}}>
// CHECK:       %[[RED:.+]]  = "onnx.ReduceMean"(%[[PRE]], %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x150x1x768x{{.*}}>, tensor<1xi64>) -> tensor<1x150x1x1x{{.*}}>
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]]) {{.*}} : (tensor<1x150x1x1x{{.*}}>, tensor<2xi64>) -> tensor<150x1x{{.*}}>
// CHECK:       return %[[POST]]
func.func @rank3_negative_axis_keepdims0(%arg0: tensor<150x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>) -> tensor<150x1x!quant.uniform<u16:f32, 7.842E-3:40316>> {
  %0 = onnx.Constant dense<[-1]> : tensor<1xi64>
  %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<150x1x768x!quant.uniform<u16:f32, 2.7333E-4:43382>>, tensor<1xi64>) -> tensor<150x1x!quant.uniform<u16:f32, 7.842E-3:40316>>
  return %1 : tensor<150x1x!quant.uniform<u16:f32, 7.842E-3:40316>>
}

// -----

// Branch 5: rank>4 + last-axis fallback, axes=[-1] keepdims=1.  Without
// normalisation the `axes[0] == rank-1` guard would compare -1 to 4 and
// skip the rewrite silently; with the fix this collapses identically to
// @rank5_collapse_middle_dims.
// CHECK-LABEL: @rank5_collapse_middle_dims_negative_axis
// CHECK-DAG:   %[[PRESHAPE_C:.+]]  = onnx.Constant dense<[2, 3, 20, 6]> : tensor<4xi64>
// CHECK-DAG:   %[[NEW_AXES:.+]]    = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[2, 3, 4, 5, 1]> : tensor<5xi64>
// CHECK:       %[[PRE:.+]]  = "onnx.Reshape"(%arg0, %[[PRESHAPE_C]])
// CHECK:       %[[RED:.+]]  = "onnx.ReduceMin"(%[[PRE]], %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]])
// CHECK:       return %[[POST]]
func.func @rank5_collapse_middle_dims_negative_axis(%arg0: tensor<2x3x4x5x6x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<2x3x4x5x1x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[-1]> : tensor<1xi64>
  %1 = "onnx.ReduceMin"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4x5x6x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<2x3x4x5x1x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<2x3x4x5x1x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Branch 4: rank>4 + leading-1 + last-axis in {2,3}, axes=[-3] keepdims=1.
// -3 normalises to 2 (rank-3) on rank=5; equivalent to
// @rank5_leading1_lastaxis_in_2_3.
// CHECK-LABEL: @rank5_leading1_negative_axis
// CHECK-DAG:   %[[PRESHAPE_C:.+]]  = onnx.Constant dense<[2, 4, 8, 16]> : tensor<4xi64>
// CHECK-DAG:   %[[NEW_AXES:.+]]    = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:   %[[POSTSHAPE_C:.+]] = onnx.Constant dense<[1, 2, 1, 8, 16]> : tensor<5xi64>
// CHECK:       %[[PRE:.+]]  = "onnx.Reshape"(%arg0, %[[PRESHAPE_C]])
// CHECK:       %[[RED:.+]]  = "onnx.ReduceMax"(%[[PRE]], %[[NEW_AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[RED]], %[[POSTSHAPE_C]])
// CHECK:       return %[[POST]]
func.func @rank5_leading1_negative_axis(%arg0: tensor<1x2x4x8x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x2x1x8x16x!quant.uniform<i8:f32, 0.05:0>> {
  %0 = onnx.Constant dense<[-3]> : tensor<1xi64>
  %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x4x8x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x2x1x8x16x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<1x2x1x8x16x!quant.uniform<i8:f32, 0.05:0>>
}
