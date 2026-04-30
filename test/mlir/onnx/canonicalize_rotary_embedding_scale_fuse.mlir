// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive: trailing single-element scalar Mul after onnx.RotaryEmbedding.
//


// Q-side: Scale directly downstream of RoPE, scale on RHS, f32, tensor<1xf32>.
func.func @rope_scale_fuse_q_f32(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xf32>, tensor<1xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_q_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_0_]], [[VAR_2_]]) : (tensor<1x16x4xf32>, tensor<1xf32>) -> tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_1_]], [[VAR_2_]]) : (tensor<1x16x4xf32>, tensor<1xf32>) -> tensor<1x16x4xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.RotaryEmbedding"([[PARAM_0_]], [[VAR_4_]], [[VAR_5_]], [[VAR_3_]]) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x3x16x8xf32>
// CHECK:         }
}

// -----

// Scale on the LHS of the Mul: NormalizeMulPattern swaps operands so the
// fuse pattern fires on a later canonicalize iteration.
func.func @rope_scale_fuse_lhs_scale(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %y     = "onnx.Mul"(%scale, %rope) : (tensor<1xf32>, tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_lhs_scale
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_0_]], [[VAR_2_]]) : (tensor<1x16x4xf32>, tensor<1xf32>) -> tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_1_]], [[VAR_2_]]) : (tensor<1x16x4xf32>, tensor<1xf32>) -> tensor<1x16x4xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.RotaryEmbedding"([[PARAM_0_]], [[VAR_4_]], [[VAR_5_]], [[VAR_3_]]) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x3x16x8xf32>
// CHECK:         }
}

// -----

// Rank-0 scalar tensor<f32> as the scale.
func.func @rope_scale_fuse_rank0_scale(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<f32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xf32>, tensor<f32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_rank0_scale
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<5.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_0_]], [[VAR_2_]]) : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_1_]], [[VAR_2_]]) : (tensor<1x16x4xf32>, tensor<f32>) -> tensor<1x16x4xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.RotaryEmbedding"([[PARAM_0_]], [[VAR_4_]], [[VAR_5_]], [[VAR_3_]]) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x3x16x8xf32>
// CHECK:         }
}

// -----

// bf16 end-to-end.
func.func @rope_scale_fuse_bf16(%X: tensor<1x3x16x8xbf16>) -> tensor<1x3x16x8xbf16> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xbf16>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xbf16>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xbf16>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xbf16>, tensor<1x16x4xbf16>, tensor<1x16x4xbf16>, none) -> tensor<1x3x16x8xbf16>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xbf16>, tensor<1xbf16>) -> tensor<1x3x16x8xbf16>
  return %y : tensor<1x3x16x8xbf16>

// CHECK-LABEL:  func.func @rope_scale_fuse_bf16
// CHECK:           [[VAR_OUT_:%.+]] = "onnx.RotaryEmbedding"({{.*}}) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xbf16>, tensor<1x16x4xbf16>, tensor<1x16x4xbf16>, none) -> tensor<1x3x16x8xbf16>
// CHECK-NOT:       "onnx.Mul"
// CHECK:      return [[VAR_OUT_]] : tensor<1x3x16x8xbf16>
}

// -----

// f16 end-to-end.
func.func @rope_scale_fuse_f16(%X: tensor<1x3x16x8xf16>) -> tensor<1x3x16x8xf16> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf16>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf16>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf16>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf16>, tensor<1x16x4xf16>, tensor<1x16x4xf16>, none) -> tensor<1x3x16x8xf16>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xf16>, tensor<1xf16>) -> tensor<1x3x16x8xf16>
  return %y : tensor<1x3x16x8xf16>

// CHECK-LABEL:  func.func @rope_scale_fuse_f16
// CHECK:           [[VAR_OUT_:%.+]] = "onnx.RotaryEmbedding"({{.*}}) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf16>, tensor<1x16x4xf16>, tensor<1x16x4xf16>, none) -> tensor<1x3x16x8xf16>
// CHECK-NOT:       "onnx.Mul"
// CHECK:      return [[VAR_OUT_]] : tensor<1x3x16x8xf16>
}

// -----

func.func @rope_scale_fuse_full_shape(%X: tensor<1x3x1601x64xf32>) -> tensor<1x3x1601x64xf32> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x1601x32xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x1601x32xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x1601x64xf32>, tensor<1x1601x32xf32>, tensor<1x1601x32xf32>, none) -> tensor<1x3x1601x64xf32>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x1601x64xf32>, tensor<1xf32>) -> tensor<1x3x1601x64xf32>
  return %y : tensor<1x3x1601x64xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_full_shape
// CHECK:           [[VAR_OUT_:%.+]] = "onnx.RotaryEmbedding"({{.*}}) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x1601x64xf32>, tensor<1x1601x32xf32>, tensor<1x1601x32xf32>, none) -> tensor<1x3x1601x64xf32>
// CHECK-NOT:       "onnx.Mul"
// CHECK:      return [[VAR_OUT_]] : tensor<1x3x1601x64xf32>
}

// -----

func.func @rope_scale_fuse_k_one_transpose(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x8x16xf32> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %t     = "onnx.Transpose"(%rope) {perm = [0, 1, 3, 2]} : (tensor<1x3x16x8xf32>) -> tensor<1x3x8x16xf32>
  %y     = "onnx.Mul"(%t, %scale) : (tensor<1x3x8x16xf32>, tensor<1xf32>) -> tensor<1x3x8x16xf32>
  return %y : tensor<1x3x8x16xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_k_one_transpose
// CHECK:           [[VAR_ROPE_:%.+]] = "onnx.RotaryEmbedding"({{.*}}) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK:      [[VAR_T_:%.+]] = "onnx.Transpose"([[VAR_ROPE_]]) {perm = [0, 1, 3, 2]} : (tensor<1x3x16x8xf32>) -> tensor<1x3x8x16xf32>
// CHECK-NOT:       "onnx.Mul"
// CHECK:      return [[VAR_T_]] : tensor<1x3x8x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative cases.

// Non-scalar scale (per-channel along the last axis): pattern must not fire.
func.func @rope_scale_fuse_neg_per_channel(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<8xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xf32>, tensor<8xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_neg_per_channel
// CHECK:           [[VAR_R_:%.+]] = "onnx.RotaryEmbedding"
// CHECK:           [[VAR_M_:%.+]] = "onnx.Mul"([[VAR_R_]], {{.*}}) : (tensor<1x3x16x8xf32>, tensor<8xf32>) -> tensor<1x3x16x8xf32>
// CHECK:           return [[VAR_M_]]
}

// -----

// Scale not a constant (function argument): pattern must not fire.
func.func @rope_scale_fuse_neg_dynamic_scale(%X: tensor<1x3x16x8xf32>, %scale: tensor<1xf32>) -> tensor<1x3x16x8xf32> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xf32>, tensor<1xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_neg_dynamic_scale
// CHECK:           [[VAR_R_:%.+]] = "onnx.RotaryEmbedding"
// CHECK:           [[VAR_M_:%.+]] = "onnx.Mul"([[VAR_R_]], %arg1)
// CHECK:           return [[VAR_M_]]
}

// -----

// RoPE has more than one use
func.func @rope_scale_fuse_neg_multi_use_rope(%X: tensor<1x3x16x8xf32>) -> (tensor<1x3x16x8xf32>, tensor<1x3x16x8xf32>) {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xf32>, tensor<1xf32>) -> tensor<1x3x16x8xf32>
  return %y, %rope : tensor<1x3x16x8xf32>, tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_neg_multi_use_rope
// CHECK:           [[VAR_R_:%.+]] = "onnx.RotaryEmbedding"
// CHECK:           [[VAR_M_:%.+]] = "onnx.Mul"([[VAR_R_]], {{.*}})
// CHECK:           return [[VAR_M_]], [[VAR_R_]]
}

// -----

// cos/sin not dense constants (function arguments): pattern must not fire.
func.func @rope_scale_fuse_neg_dynamic_cos_sin(%X: tensor<1x3x16x8xf32>, %cos: tensor<1x16x4xf32>, %sin: tensor<1x16x4xf32>) -> tensor<1x3x16x8xf32> {
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xf32>, tensor<1xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_neg_dynamic_cos_sin
// CHECK:           [[VAR_R_:%.+]] = "onnx.RotaryEmbedding"(%arg0, %arg1, %arg2,
// CHECK:           [[VAR_M_:%.+]] = "onnx.Mul"([[VAR_R_]], {{.*}})
// CHECK:           return [[VAR_M_]]
}

// -----

// interleaved=1
func.func @rope_scale_fuse_neg_interleaved(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 1 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xf32>, tensor<1xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_neg_interleaved
// CHECK:           [[VAR_R_:%.+]] = "onnx.RotaryEmbedding"({{.*}}) {interleaved = 1 : si64
// CHECK:           [[VAR_M_:%.+]] = "onnx.Mul"([[VAR_R_]], {{.*}})
// CHECK:           return [[VAR_M_]]
}

// -----

// rotary_embedding_dim != 0 (partial rotation)
func.func @rope_scale_fuse_neg_partial_rotation(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x2xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x2xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 4 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x2xf32>, tensor<1x16x2xf32>, none) -> tensor<1x3x16x8xf32>
  %y     = "onnx.Mul"(%rope, %scale) : (tensor<1x3x16x8xf32>, tensor<1xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_neg_partial_rotation
// CHECK:           [[VAR_R_:%.+]] = "onnx.RotaryEmbedding"({{.*}}) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 4 : si64
// CHECK:           [[VAR_M_:%.+]] = "onnx.Mul"([[VAR_R_]], {{.*}})
// CHECK:           return [[VAR_M_]]
}

// -----

// A Transpose on the chain has more than one use
func.func @rope_scale_fuse_neg_multi_use_transpose(%X: tensor<1x3x16x8xf32>) -> (tensor<1x3x8x16xf32>, tensor<1x3x8x16xf32>) {
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %rope  = "onnx.RotaryEmbedding"(%X, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
  %t     = "onnx.Transpose"(%rope) {perm = [0, 1, 3, 2]} : (tensor<1x3x16x8xf32>) -> tensor<1x3x8x16xf32>
  %y     = "onnx.Mul"(%t, %scale) : (tensor<1x3x8x16xf32>, tensor<1xf32>) -> tensor<1x3x8x16xf32>
  return %y, %t : tensor<1x3x8x16xf32>, tensor<1x3x8x16xf32>

// CHECK-LABEL:  func.func @rope_scale_fuse_neg_multi_use_transpose
// CHECK:           "onnx.RotaryEmbedding"
// CHECK:           [[VAR_T_:%.+]] = "onnx.Transpose"
// CHECK:           [[VAR_M_:%.+]] = "onnx.Mul"([[VAR_T_]], {{.*}})
// CHECK:           return [[VAR_M_]], [[VAR_T_]]
}
