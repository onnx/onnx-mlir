// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive: prefix slice/concat sandwich elimination around onnx.RotaryEmbedding.


// Q-side, no transposes around RoPE.
func.func @rope_q_prefix_carveout(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s1   = onnx.Constant dense<1> : tensor<1xi64>
  %s16  = onnx.Constant dense<16> : tensor<1xi64>
  %ax   = onnx.Constant dense<2> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre  = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat  = "onnx.Slice"(%X, %s1, %s16, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %y    = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_q_prefix_carveout
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
// CHECK-DAG:       [[VAR_NONE_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_COS_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
// CHECK-DAG:       [[VAR_SIN_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
// CHECK-DAG:       [[VAR_COSID_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1x1x4xf32>
// CHECK-DAG:       [[VAR_SINID_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1x1x4xf32>
// CHECK-DAG:       [[VAR_PCOS_:%.+]] = "onnx.Concat"([[VAR_COSID_]], [[VAR_COS_]]) {axis = 1 : si64} : (tensor<1x1x4xf32>, tensor<1x15x4xf32>) -> tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_PSIN_:%.+]] = "onnx.Concat"([[VAR_SINID_]], [[VAR_SIN_]]) {axis = 1 : si64} : (tensor<1x1x4xf32>, tensor<1x15x4xf32>) -> tensor<1x16x4xf32>
// CHECK:           [[VAR_OUT_:%.+]] = "onnx.RotaryEmbedding"([[PARAM_0_]], [[VAR_PCOS_]], [[VAR_PSIN_]], [[VAR_NONE_]]) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK:           return [[VAR_OUT_]] : tensor<1x3x16x8xf32>
// CHECK-NOT:       onnx.Slice
}

// -----

// K-side: transpose pair around RoPE.
func.func @rope_k_prefix_carveout(%X: tensor<1x16x3x8xf32>) -> tensor<1x16x3x8xf32> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s1   = onnx.Constant dense<1> : tensor<1xi64>
  %s16  = onnx.Constant dense<16> : tensor<1xi64>
  %ax   = onnx.Constant dense<1> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre  = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x16x3x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x3x8xf32>
  %pat  = "onnx.Slice"(%X, %s1, %s16, %ax, %st) : (tensor<1x16x3x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x15x3x8xf32>
  %tp1  = "onnx.Transpose"(%pat) {perm = [0, 2, 1, 3]} : (tensor<1x15x3x8xf32>) -> tensor<1x3x15x8xf32>
  %rope = "onnx.RotaryEmbedding"(%tp1, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %tp2  = "onnx.Transpose"(%rope) {perm = [0, 2, 1, 3]} : (tensor<1x3x15x8xf32>) -> tensor<1x15x3x8xf32>
  %y    = "onnx.Concat"(%pre, %tp2) {axis = 1 : si64} : (tensor<1x1x3x8xf32>, tensor<1x15x3x8xf32>) -> tensor<1x16x3x8xf32>
  return %y : tensor<1x16x3x8xf32>

// CHECK-LABEL:  func.func @rope_k_prefix_carveout
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x16x3x8xf32>) -> tensor<1x16x3x8xf32> {
// CHECK-DAG:       [[VAR_NONE_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_SINID_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1x1x4xf32>
// CHECK-DAG:       [[VAR_COSID_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1x1x4xf32>
// CHECK-DAG:       [[VAR_COS_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
// CHECK-DAG:       [[VAR_SIN_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
// CHECK-DAG:       [[VAR_TPRE_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 1, 3]} : (tensor<1x16x3x8xf32>) -> tensor<1x3x16x8xf32>
// CHECK-DAG:       [[VAR_PCOS_:%.+]] = "onnx.Concat"([[VAR_COSID_]], [[VAR_COS_]]) {axis = 1 : si64} : (tensor<1x1x4xf32>, tensor<1x15x4xf32>) -> tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_PSIN_:%.+]] = "onnx.Concat"([[VAR_SINID_]], [[VAR_SIN_]]) {axis = 1 : si64} : (tensor<1x1x4xf32>, tensor<1x15x4xf32>) -> tensor<1x16x4xf32>
// CHECK:           [[VAR_ROPE_:%.+]] = "onnx.RotaryEmbedding"([[VAR_TPRE_]], [[VAR_PCOS_]], [[VAR_PSIN_]], [[VAR_NONE_]]) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK:           [[VAR_TPOST_:%.+]] = "onnx.Transpose"([[VAR_ROPE_]]) {perm = [0, 2, 1, 3]} : (tensor<1x3x16x8xf32>) -> tensor<1x16x3x8xf32>
// CHECK:           return [[VAR_TPOST_]] : tensor<1x16x3x8xf32>
// CHECK-NOT:       onnx.Slice
}

// -----

// bf16 element type.
func.func @rope_q_bf16(%X: tensor<1x3x16x8xbf16>) -> tensor<1x3x16x8xbf16> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s1   = onnx.Constant dense<1> : tensor<1xi64>
  %s16  = onnx.Constant dense<16> : tensor<1xi64>
  %ax   = onnx.Constant dense<2> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xbf16>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xbf16>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre  = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xbf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xbf16>
  %pat  = "onnx.Slice"(%X, %s1, %s16, %ax, %st) : (tensor<1x3x16x8xbf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xbf16>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xbf16>, tensor<1x15x4xbf16>, tensor<1x15x4xbf16>, none) -> tensor<1x3x15x8xbf16>
  %y    = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x8xbf16>, tensor<1x3x15x8xbf16>) -> tensor<1x3x16x8xbf16>
  return %y : tensor<1x3x16x8xbf16>

// CHECK-LABEL:  func.func @rope_q_bf16
// CHECK-DAG:       [[VAR_COSID_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1x1x4xbf16>
// CHECK-DAG:       [[VAR_SINID_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1x1x4xbf16>
// CHECK:           "onnx.RotaryEmbedding"({{.*}}) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xbf16>, tensor<1x16x4xbf16>, tensor<1x16x4xbf16>, none) -> tensor<1x3x16x8xbf16>
// CHECK-NOT:       onnx.Slice
}

// -----

// f16 element type.
func.func @rope_q_f16(%X: tensor<1x3x16x8xf16>) -> tensor<1x3x16x8xf16> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s1   = onnx.Constant dense<1> : tensor<1xi64>
  %s16  = onnx.Constant dense<16> : tensor<1xi64>
  %ax   = onnx.Constant dense<2> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf16>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf16>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre  = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf16>
  %pat  = "onnx.Slice"(%X, %s1, %s16, %ax, %st) : (tensor<1x3x16x8xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf16>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf16>, tensor<1x15x4xf16>, tensor<1x15x4xf16>, none) -> tensor<1x3x15x8xf16>
  %y    = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x8xf16>, tensor<1x3x15x8xf16>) -> tensor<1x3x16x8xf16>
  return %y : tensor<1x3x16x8xf16>

// CHECK-LABEL:  func.func @rope_q_f16
// CHECK-DAG:       [[VAR_COSID_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1x1x4xf16>
// CHECK-DAG:       [[VAR_SINID_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1x1x4xf16>
// CHECK:           "onnx.RotaryEmbedding"({{.*}}) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf16>, tensor<1x16x4xf16>, tensor<1x16x4xf16>, none) -> tensor<1x3x16x8xf16>
// CHECK-NOT:       onnx.Slice
}

// -----

func.func @rope_q_negative_axis(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s1   = onnx.Constant dense<1> : tensor<1xi64>
  %s16  = onnx.Constant dense<16> : tensor<1xi64>
  %ax   = onnx.Constant dense<2> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre  = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat  = "onnx.Slice"(%X, %s1, %s16, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %y    = "onnx.Concat"(%pre, %rope) {axis = -2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_q_negative_axis
// CHECK:           "onnx.RotaryEmbedding"({{.*}}) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK-NOT:       onnx.Slice
}

// -----

func.func @rope_q_large_shapes(%X: tensor<1x3x1601x64xf32>) -> tensor<1x3x1601x64xf32> {
  %s0    = onnx.Constant dense<0> : tensor<1xi64>
  %s1    = onnx.Constant dense<1> : tensor<1xi64>
  %s1601 = onnx.Constant dense<1601> : tensor<1xi64>
  %ax    = onnx.Constant dense<2> : tensor<1xi64>
  %st    = onnx.Constant dense<1> : tensor<1xi64>
  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x1600x32xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x1600x32xf32>
  %none  = "onnx.NoValue"() {value} : () -> none
  %pre   = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x1601x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x64xf32>
  %pat   = "onnx.Slice"(%X, %s1, %s1601, %ax, %st) : (tensor<1x3x1601x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1600x64xf32>
  %rope  = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x1600x64xf32>, tensor<1x1600x32xf32>, tensor<1x1600x32xf32>, none) -> tensor<1x3x1600x64xf32>
  %y     = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x64xf32>, tensor<1x3x1600x64xf32>) -> tensor<1x3x1601x64xf32>
  return %y : tensor<1x3x1601x64xf32>

// CHECK-LABEL:  func.func @rope_q_large_shapes
// CHECK-DAG:       [[VAR_COSID_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1x1x32xf32>
// CHECK-DAG:       [[VAR_SINID_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1x1x32xf32>
// CHECK:           "onnx.RotaryEmbedding"({{.*}}) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x1601x64xf32>, tensor<1x1601x32xf32>, tensor<1x1601x32xf32>, none) -> tensor<1x3x1601x64xf32>
// CHECK-NOT:       onnx.Slice
}

// -----

// Shared cos/sin: Q (axis 2, no transpose) and K (axis 1, transpose pair)
func.func @rope_qk_shared_cos_sin(%Q: tensor<1x3x16x8xf32>, %K: tensor<1x16x3x8xf32>) -> (tensor<1x3x16x8xf32>, tensor<1x16x3x8xf32>) {
  %s0 = onnx.Constant dense<0> : tensor<1xi64>
  %s1 = onnx.Constant dense<1> : tensor<1xi64>
  %s16 = onnx.Constant dense<16> : tensor<1xi64>
  %ax2 = onnx.Constant dense<2> : tensor<1xi64>
  %ax1 = onnx.Constant dense<1> : tensor<1xi64>
  %st = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
  %sin = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none

  %qpre = "onnx.Slice"(%Q, %s0, %s1, %ax2, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %qpat = "onnx.Slice"(%Q, %s1, %s16, %ax2, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>
  %qrope = "onnx.RotaryEmbedding"(%qpat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %qy = "onnx.Concat"(%qpre, %qrope) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>

  %kpre = "onnx.Slice"(%K, %s0, %s1, %ax1, %st) : (tensor<1x16x3x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x3x8xf32>
  %kpat = "onnx.Slice"(%K, %s1, %s16, %ax1, %st) : (tensor<1x16x3x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x15x3x8xf32>
  %ktp1 = "onnx.Transpose"(%kpat) {perm = [0, 2, 1, 3]} : (tensor<1x15x3x8xf32>) -> tensor<1x3x15x8xf32>
  %krope = "onnx.RotaryEmbedding"(%ktp1, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %ktp2 = "onnx.Transpose"(%krope) {perm = [0, 2, 1, 3]} : (tensor<1x3x15x8xf32>) -> tensor<1x15x3x8xf32>
  %ky = "onnx.Concat"(%kpre, %ktp2) {axis = 1 : si64} : (tensor<1x1x3x8xf32>, tensor<1x15x3x8xf32>) -> tensor<1x16x3x8xf32>

  return %qy, %ky : tensor<1x3x16x8xf32>, tensor<1x16x3x8xf32>

// CHECK-LABEL:  func.func @rope_qk_shared_cos_sin
// CHECK-COUNT-2:   "onnx.RotaryEmbedding"
// CHECK-NOT:       onnx.Slice
}

// -----

func.func @rope_q_two_layers(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s1   = onnx.Constant dense<1> : tensor<1xi64>
  %s16  = onnx.Constant dense<16> : tensor<1xi64>
  %ax   = onnx.Constant dense<2> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none

  %pre0  = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat0  = "onnx.Slice"(%X, %s1, %s16, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>
  %rope0 = "onnx.RotaryEmbedding"(%pat0, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %y0    = "onnx.Concat"(%pre0, %rope0) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>

  %pre1  = "onnx.Slice"(%y0, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat1  = "onnx.Slice"(%y0, %s1, %s16, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>
  %rope1 = "onnx.RotaryEmbedding"(%pat1, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %y1    = "onnx.Concat"(%pre1, %rope1) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>

  return %y1 : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_q_two_layers
// CHECK-COUNT-2:   "onnx.RotaryEmbedding"
// CHECK-NOT:       onnx.Slice
}

// -----

// prefix slice carves 2 tokens (ends=2) but prefixLen == 1; must not fire.
func.func @neg_prefix_ends_two(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s2   = onnx.Constant dense<2> : tensor<1xi64>
  %s16  = onnx.Constant dense<16> : tensor<1xi64>
  %ax   = onnx.Constant dense<2> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x14x4xf32>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x14x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre  = "onnx.Slice"(%X, %s0, %s2, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x2x8xf32>
  %pat  = "onnx.Slice"(%X, %s2, %s16, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x14x8xf32>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x14x8xf32>, tensor<1x14x4xf32>, tensor<1x14x4xf32>, none) -> tensor<1x3x14x8xf32>
  %y    = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x2x8xf32>, tensor<1x3x14x8xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @neg_prefix_ends_two
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.RotaryEmbedding"
// CHECK:           "onnx.Concat"
}

// -----

// Patches slice has starts=[2] (gap between prefix and patches); must not fire.
func.func @neg_patches_starts_two(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x15x8xf32> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s1   = onnx.Constant dense<1> : tensor<1xi64>
  %s2   = onnx.Constant dense<2> : tensor<1xi64>
  %s16  = onnx.Constant dense<16> : tensor<1xi64>
  %ax   = onnx.Constant dense<2> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x14x4xf32>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x14x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre  = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat  = "onnx.Slice"(%X, %s2, %s16, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x14x8xf32>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x14x8xf32>, tensor<1x14x4xf32>, tensor<1x14x4xf32>, none) -> tensor<1x3x14x8xf32>
  %y    = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x14x8xf32>) -> tensor<1x3x15x8xf32>
  return %y : tensor<1x3x15x8xf32>

// CHECK-LABEL:  func.func @neg_patches_starts_two
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.RotaryEmbedding"
}

// -----

// prefix and patches slice read different SSA sources; must not fire.
func.func @neg_different_sources(%X: tensor<1x3x16x8xf32>, %Y: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s1   = onnx.Constant dense<1> : tensor<1xi64>
  %s16  = onnx.Constant dense<16> : tensor<1xi64>
  %ax   = onnx.Constant dense<2> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre  = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat  = "onnx.Slice"(%Y, %s1, %s16, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %y    = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @neg_different_sources
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.RotaryEmbedding"
// CHECK:           "onnx.Concat"
}

// -----

// position_ids is non-NoValue; must not fire.
func.func @neg_position_ids(%X: tensor<1x3x16x8xf32>, %P: tensor<1x16xi64>) -> tensor<1x3x16x8xf32> {
  %s0 = onnx.Constant dense<0> : tensor<1xi64>
  %s1 = onnx.Constant dense<1> : tensor<1xi64>
  %s16 = onnx.Constant dense<16> : tensor<1xi64>
  %ax = onnx.Constant dense<2> : tensor<1xi64>
  %st = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
  %sin = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
  %pre = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat = "onnx.Slice"(%X, %s1, %s16, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %P) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, tensor<1x16xi64>) -> tensor<1x3x15x8xf32>
  %y = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @neg_position_ids
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.RotaryEmbedding"
// CHECK:           "onnx.Concat"
}

// -----

// interleaved=1; must not fire.
func.func @neg_interleaved(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %s0 = onnx.Constant dense<0> : tensor<1xi64>
  %s1 = onnx.Constant dense<1> : tensor<1xi64>
  %s16 = onnx.Constant dense<16> : tensor<1xi64>
  %ax = onnx.Constant dense<2> : tensor<1xi64>
  %st = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
  %sin = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre = "onnx.Slice"(%X, %s0, %s1, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat = "onnx.Slice"(%X, %s1, %s16, %ax, %st) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 1 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %y = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @neg_interleaved
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.RotaryEmbedding"
// CHECK-SAME:      interleaved = 1
// CHECK:           "onnx.Concat"
}

// -----

// prefix slice has steps=[2]; must not fire.
func.func @neg_step_two(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %s0 = onnx.Constant dense<0> : tensor<1xi64>
  %s1 = onnx.Constant dense<1> : tensor<1xi64>
  %s2 = onnx.Constant dense<2> : tensor<1xi64>
  %s16 = onnx.Constant dense<16> : tensor<1xi64>
  %ax = onnx.Constant dense<2> : tensor<1xi64>
  %cos = onnx.Constant dense<2.000000e+00> : tensor<1x15x4xf32>
  %sin = onnx.Constant dense<3.000000e+00> : tensor<1x15x4xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre = "onnx.Slice"(%X, %s0, %s1, %ax, %s2) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat = "onnx.Slice"(%X, %s1, %s16, %ax, %s1) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>
  %rope = "onnx.RotaryEmbedding"(%pat, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x15x8xf32>, tensor<1x15x4xf32>, tensor<1x15x4xf32>, none) -> tensor<1x3x15x8xf32>
  %y = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>
  return %y : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @neg_step_two
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.RotaryEmbedding"
// CHECK:           "onnx.Concat"
}

// -----

// Pre-RoPE Transpose chain does not map the carve axis to the RoPE seq
// axis (axis 2). X here has two axes of equal size (axes 2 and 3, both
// size 4 after the patches slice), 
func.func @neg_perm_wrong_axis(%X: tensor<1x5x5x4xf32>) -> tensor<1x5x5x4xf32> {
  %s0   = onnx.Constant dense<0> : tensor<1xi64>
  %s1   = onnx.Constant dense<1> : tensor<1xi64>
  %s5   = onnx.Constant dense<5> : tensor<1xi64>
  %ax2  = onnx.Constant dense<2> : tensor<1xi64>
  %st   = onnx.Constant dense<1> : tensor<1xi64>
  %cos  = onnx.Constant dense<2.000000e+00> : tensor<1x4x2xf32>
  %sin  = onnx.Constant dense<3.000000e+00> : tensor<1x4x2xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %pre  = "onnx.Slice"(%X, %s0, %s1, %ax2, %st) : (tensor<1x5x5x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x5x1x4xf32>
  %pat  = "onnx.Slice"(%X, %s1, %s5, %ax2, %st) : (tensor<1x5x5x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x5x4x4xf32>
  %tp   = "onnx.Transpose"(%pat) {perm = [0, 1, 3, 2]} : (tensor<1x5x4x4xf32>) -> tensor<1x5x4x4xf32>
  %rope = "onnx.RotaryEmbedding"(%tp, %cos, %sin, %none) {interleaved = 0 : si64, num_heads = 5 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x5x4x4xf32>, tensor<1x4x2xf32>, tensor<1x4x2xf32>, none) -> tensor<1x5x4x4xf32>
  %tp2  = "onnx.Transpose"(%rope) {perm = [0, 1, 3, 2]} : (tensor<1x5x4x4xf32>) -> tensor<1x5x4x4xf32>
  %y    = "onnx.Concat"(%pre, %tp2) {axis = 2 : si64} : (tensor<1x5x1x4xf32>, tensor<1x5x4x4xf32>) -> tensor<1x5x5x4xf32>
  return %y : tensor<1x5x5x4xf32>

// CHECK-LABEL:  func.func @neg_perm_wrong_axis
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Slice"
// CHECK:           "onnx.Transpose"
// CHECK:           "onnx.RotaryEmbedding"
// CHECK:           "onnx.Transpose"
// CHECK:           "onnx.Concat"
}
