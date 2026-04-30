// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --recompose-onnx="enable-rotary-embedding-recompose=true" --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s -split-input-file | FileCheck %s --check-prefix=NOFLAG

// -----

// Q-side layout: patches is [B, N, S, D]. cos/sin broadcast on the head axis
// (`1x1xSxD`). 

func.func @rope_q_side(%patches: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x1x16x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x16x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<1x3x16x4xf32>) -> tensor<1x3x16x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x16x4xf32>, tensor<1x3x16x4xf32>) -> tensor<1x3x16x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x16x8xf32>, tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32>
  return %out : tensor<1x3x16x8xf32>

// CHECK-LABEL: func.func @rope_q_side
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
// CHECK-DAG:     [[VAR_COS_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:     [[VAR_SIN_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:     [[VAR_NONE_:%.+]] = "onnx.NoValue"
// CHECK:         [[VAR_OUT_:%.+]] = "onnx.RotaryEmbedding"([[PARAM_0_]], [[VAR_COS_]], [[VAR_SIN_]], [[VAR_NONE_]]) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK:         return [[VAR_OUT_]] : tensor<1x3x16x8xf32>
// CHECK-NOT:     onnx.Concat
// CHECK-NOT:     onnx.Neg
// CHECK-NOT:     onnx.Slice
// CHECK-NOT:     onnx.Transpose

// NOFLAG-LABEL: func.func @rope_q_side
// NOFLAG-NOT:   onnx.RotaryEmbedding
// NOFLAG:       onnx.Add
}

// -----

// K-side layout: patches is [B, S, N, D]. cos/sin broadcast on the head axis
// (`1xSx1xD`). The rewrite inserts a Transpose(perm=[0,2,1,3]) before the op
// to reach the [B,N,S,D] contract and a matching Transpose after to restore
// the original layout.

func.func @rope_k_side(%patches: tensor<1x16x3x8xf32>) -> tensor<1x16x3x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x16x1x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x16x1x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x16x3x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x16x3x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x16x3x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x16x3x4xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<1x16x3x4xf32>) -> tensor<1x16x3x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x16x3x4xf32>, tensor<1x16x3x4xf32>) -> tensor<1x16x3x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x16x3x8xf32>, tensor<1x16x1x8xf32>) -> tensor<1x16x3x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x16x3x8xf32>, tensor<1x16x1x8xf32>) -> tensor<1x16x3x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x16x3x8xf32>, tensor<1x16x3x8xf32>) -> tensor<1x16x3x8xf32>
  return %out : tensor<1x16x3x8xf32>

// CHECK-LABEL: func.func @rope_k_side
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<1x16x3x8xf32>) -> tensor<1x16x3x8xf32> {
// CHECK-DAG:     [[VAR_COS_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:     [[VAR_SIN_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf32>
// CHECK-DAG:     [[VAR_NONE_:%.+]] = "onnx.NoValue"
// CHECK:         [[VAR_T1_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 1, 3]} : (tensor<1x16x3x8xf32>) -> tensor<1x3x16x8xf32>
// CHECK:         [[VAR_R_:%.+]] = "onnx.RotaryEmbedding"([[VAR_T1_]], [[VAR_COS_]], [[VAR_SIN_]], [[VAR_NONE_]]) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK:         [[VAR_T2_:%.+]] = "onnx.Transpose"([[VAR_R_]]) {perm = [0, 2, 1, 3]} : (tensor<1x3x16x8xf32>) -> tensor<1x16x3x8xf32>
// CHECK:         return [[VAR_T2_]] : tensor<1x16x3x8xf32>
// CHECK-NOT:     onnx.Concat
// CHECK-NOT:     onnx.Neg
// CHECK-NOT:     onnx.Slice
}

// -----

// K-side layout where N == S: use the cos/sin broadcast axis, not the equal
// dimension sizes, to classify the layout.

func.func @rope_k_side_equal_n_s(%patches: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x4x1x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x4x1x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x4x4x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x4x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x4x4x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x4x4xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x4x4x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x4x4x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x4x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x4x4x8xf32>, tensor<1x4x1x8xf32>) -> tensor<1x4x4x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x4x4x8xf32>, tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  return %out : tensor<1x4x4x8xf32>

// CHECK-LABEL: func.func @rope_k_side_equal_n_s
// CHECK-SAME:  ([[PARAM_0_:%.+]]: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32> {
// CHECK-DAG:     [[VAR_COS_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x4x4xf32>
// CHECK-DAG:     [[VAR_SIN_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1x4x4xf32>
// CHECK-DAG:     [[VAR_NONE_:%.+]] = "onnx.NoValue"
// CHECK:         [[VAR_T1_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 1, 3]} : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
// CHECK:         [[VAR_R_:%.+]] = "onnx.RotaryEmbedding"([[VAR_T1_]], [[VAR_COS_]], [[VAR_SIN_]], [[VAR_NONE_]]) {interleaved = 0 : si64, num_heads = 4 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x4x4x8xf32>, tensor<1x4x4xf32>, tensor<1x4x4xf32>, none) -> tensor<1x4x4x8xf32>
// CHECK:         [[VAR_T2_:%.+]] = "onnx.Transpose"([[VAR_R_]]) {perm = [0, 2, 1, 3]} : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
// CHECK:         return [[VAR_T2_]] : tensor<1x4x4x8xf32>
}

// -----

// Non-splat cos/sin where each stripe has lo half == hi half.

func.func @rope_nonsplat_cos(%patches: tensor<1x3x2x8xf32>) -> tensor<1x3x2x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<[[[[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]]]]> : tensor<1x1x2x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x2x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x2x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x2x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x2x4xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<1x3x2x4xf32>) -> tensor<1x3x2x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x2x4xf32>, tensor<1x3x2x4xf32>) -> tensor<1x3x2x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x2x8xf32>, tensor<1x1x2x8xf32>) -> tensor<1x3x2x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x2x8xf32>, tensor<1x1x2x8xf32>) -> tensor<1x3x2x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x2x8xf32>, tensor<1x3x2x8xf32>) -> tensor<1x3x2x8xf32>
  return %out : tensor<1x3x2x8xf32>

// Verify the halved cos preserves the [1,2,3,4][5,6,7,8] per-row values
// AND that the surrounding op signature is exactly what we expect (op
// input == patches, halved cos/sin shapes, none position_ids, attributes,
// no leftover Slice/Neg/Concat/Mul/Add).
// CHECK-LABEL:  func.func @rope_nonsplat_cos
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x2x8xf32>) -> tensor<1x3x2x8xf32> {
// CHECK-DAG:       [[VAR_COS_:%.+]] = onnx.Constant dense<{{.}}{{.}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]{{.}}{{.}}> : tensor<1x2x4xf32>
// CHECK-DAG:       [[VAR_SIN_:%.+]] = onnx.Constant dense<3.000000e+00> : tensor<1x2x4xf32>
// CHECK-DAG:       [[VAR_NONE_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_OUT_:%.+]] = "onnx.RotaryEmbedding"([[PARAM_0_]], [[VAR_COS_]], [[VAR_SIN_]], [[VAR_NONE_]]) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x2x8xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>, none) -> tensor<1x3x2x8xf32>
// CHECK:           return [[VAR_OUT_]] : tensor<1x3x2x8xf32>
// CHECK-NOT:       onnx.Slice
// CHECK-NOT:       onnx.Neg
// CHECK-NOT:       onnx.Concat
// CHECK-NOT:       onnx.Mul
// CHECK-NOT:       onnx.Add
}

// -----

// Negative: cos/sin tables are not duplicate-half along the last axis.

func.func @rope_negative_nondup_cos(%patches: tensor<1x3x4x8xf32>) -> tensor<1x3x4x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  // lo half [1,1,1,1] != hi half [2,2,2,2]
  %cos = onnx.Constant dense<[[[[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]]]]> : tensor<1x1x4x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x4x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x4x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x4x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x4x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x4x4xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x4x8xf32>, tensor<1x1x4x8xf32>) -> tensor<1x3x4x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x4x8xf32>, tensor<1x1x4x8xf32>) -> tensor<1x3x4x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x4x8xf32>, tensor<1x3x4x8xf32>) -> tensor<1x3x4x8xf32>
  return %out : tensor<1x3x4x8xf32>

// CHECK-LABEL: func.func @rope_negative_nondup_cos
// CHECK-NOT:   onnx.RotaryEmbedding
}

// -----

// Negative: cos/sin seq axis matches neither patches[1] nor patches[2] (here
// the only non-broadcast-1 axis is the batch axis).

func.func @rope_negative_seq_axis(%patches: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  // cos has a non-1 axis 0 (batch dim 7, not used as seq) -- not a match for
  // patches[1]=3 or patches[2]=16.
  %cos = onnx.Constant dense<2.0> : tensor<7x1x1x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<7x1x1x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<1x3x16x4xf32>) -> tensor<1x3x16x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x16x4xf32>, tensor<1x3x16x4xf32>) -> tensor<1x3x16x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x16x8xf32>, tensor<7x1x1x8xf32>) -> tensor<7x3x16x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x16x8xf32>, tensor<7x1x1x8xf32>) -> tensor<7x3x16x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<7x3x16x8xf32>, tensor<7x3x16x8xf32>) -> tensor<7x3x16x8xf32>
  %ret = "onnx.ReduceSum"(%out, %starts_lo) {keepdims = 0 : si64} : (tensor<7x3x16x8xf32>, tensor<1xi64>) -> tensor<3x16x8xf32>
  %expand_shape = onnx.Constant dense<[1, 3, 16, 8]> : tensor<4xi64>
  %expanded = "onnx.Reshape"(%ret, %expand_shape) : (tensor<3x16x8xf32>, tensor<4xi64>) -> tensor<1x3x16x8xf32>
  return %expanded : tensor<1x3x16x8xf32>

// CHECK-LABEL: func.func @rope_negative_seq_axis
// CHECK-NOT:   onnx.RotaryEmbedding
}

// -----

// Negative: a Mul has multiple uses
func.func @rope_negative_multi_use(%patches: tensor<1x3x16x8xf32>) -> (tensor<1x3x16x8xf32>, tensor<1x3x16x8xf32>) {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x1x16x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x16x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<1x3x16x4xf32>) -> tensor<1x3x16x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x16x4xf32>, tensor<1x3x16x4xf32>) -> tensor<1x3x16x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x16x8xf32>, tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32>
  return %out, %m_data : tensor<1x3x16x8xf32>, tensor<1x3x16x8xf32>

// CHECK-LABEL: func.func @rope_negative_multi_use
// CHECK-NOT:   onnx.RotaryEmbedding

}

// -----

// Negative: f64 element type is outside the matcher's whitelist

func.func @rope_negative_f64(%patches: tensor<1x3x16x8xf64>) -> tensor<1x3x16x8xf64> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x1x16x8xf64>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x16x8xf64>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x16x8xf64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf64>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x16x8xf64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf64>
  %neg = "onnx.Neg"(%hi) : (tensor<1x3x16x4xf64>) -> tensor<1x3x16x4xf64>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x16x4xf64>, tensor<1x3x16x4xf64>) -> tensor<1x3x16x8xf64>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x16x8xf64>, tensor<1x1x16x8xf64>) -> tensor<1x3x16x8xf64>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x16x8xf64>, tensor<1x1x16x8xf64>) -> tensor<1x3x16x8xf64>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x16x8xf64>, tensor<1x3x16x8xf64>) -> tensor<1x3x16x8xf64>
  return %out : tensor<1x3x16x8xf64>

// CHECK-LABEL: func.func @rope_negative_f64
// CHECK-NOT:   onnx.RotaryEmbedding
}

// -----

// Positive: bf16 element type matches the whitelist.

func.func @rope_q_side_bf16(%patches: tensor<1x3x16x8xbf16>) -> tensor<1x3x16x8xbf16> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x1x16x8xbf16>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x16x8xbf16>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x16x8xbf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xbf16>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x16x8xbf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xbf16>
  %neg = "onnx.Neg"(%hi) : (tensor<1x3x16x4xbf16>) -> tensor<1x3x16x4xbf16>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x16x4xbf16>, tensor<1x3x16x4xbf16>) -> tensor<1x3x16x8xbf16>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x16x8xbf16>, tensor<1x1x16x8xbf16>) -> tensor<1x3x16x8xbf16>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x16x8xbf16>, tensor<1x1x16x8xbf16>) -> tensor<1x3x16x8xbf16>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x16x8xbf16>, tensor<1x3x16x8xbf16>) -> tensor<1x3x16x8xbf16>
  return %out : tensor<1x3x16x8xbf16>

// CHECK-LABEL: func.func @rope_q_side_bf16
// CHECK-DAG:     onnx.Constant dense<2.000000e+00> : tensor<1x16x4xbf16>
// CHECK-DAG:     onnx.Constant dense<3.000000e+00> : tensor<1x16x4xbf16>
// CHECK:         "onnx.RotaryEmbedding"

}

// -----

// Positive: f16 element type matches the whitelist.

func.func @rope_q_side_f16(%patches: tensor<1x3x16x8xf16>) -> tensor<1x3x16x8xf16> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x1x16x8xf16>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x16x8xf16>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x16x8xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf16>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x16x8xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf16>
  %neg = "onnx.Neg"(%hi) : (tensor<1x3x16x4xf16>) -> tensor<1x3x16x4xf16>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x16x4xf16>, tensor<1x3x16x4xf16>) -> tensor<1x3x16x8xf16>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x16x8xf16>, tensor<1x1x16x8xf16>) -> tensor<1x3x16x8xf16>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x16x8xf16>, tensor<1x1x16x8xf16>) -> tensor<1x3x16x8xf16>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x16x8xf16>, tensor<1x3x16x8xf16>) -> tensor<1x3x16x8xf16>
  return %out : tensor<1x3x16x8xf16>

// CHECK-LABEL: func.func @rope_q_side_f16
// CHECK-DAG:     onnx.Constant dense<2.000000e+00> : tensor<1x16x4xf16>
// CHECK-DAG:     onnx.Constant dense<3.000000e+00> : tensor<1x16x4xf16>
// CHECK:         "onnx.RotaryEmbedding"

}

// -----

// Negative: rank-3 patches are bailed for now

func.func @rope_negative_rank3(%patches: tensor<3x16x8xf32>) -> tensor<3x16x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<2> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x16x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x16x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x16x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x16x4xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<3x16x4xf32>) -> tensor<3x16x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<3x16x4xf32>, tensor<3x16x4xf32>) -> tensor<3x16x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<3x16x8xf32>, tensor<1x16x8xf32>) -> tensor<3x16x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<3x16x8xf32>, tensor<1x16x8xf32>) -> tensor<3x16x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<3x16x8xf32>, tensor<3x16x8xf32>) -> tensor<3x16x8xf32>
  return %out : tensor<3x16x8xf32>

// CHECK-LABEL: func.func @rope_negative_rank3
// CHECK-NOT:   onnx.RotaryEmbedding

}

// -----

// Negative: mirrored Concat order (`Concat(half_hi, Neg(half_lo))`) is not
// matched for now

func.func @rope_negative_mirrored_concat(%patches: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x1x16x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x16x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %neg = "onnx.Neg"(%lo) : (tensor<1x3x16x4xf32>) -> tensor<1x3x16x4xf32>
  // Mirrored: Concat(half_hi, Neg(half_lo)) instead of Concat(Neg(half_hi), half_lo).
  %rot = "onnx.Concat"(%hi, %neg) {axis = -1 : si64} : (tensor<1x3x16x4xf32>, tensor<1x3x16x4xf32>) -> tensor<1x3x16x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x16x8xf32>, tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32>
  return %out : tensor<1x3x16x8xf32>

// CHECK-LABEL: func.func @rope_negative_mirrored_concat
// CHECK-NOT:   onnx.RotaryEmbedding
}

// -----

// Negative: slice step != 1 (here step == 2) is not a valid rotate_half.

func.func @rope_negative_slice_step(%patches: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps_one = onnx.Constant dense<1> : tensor<1xi64>
  %steps_two = onnx.Constant dense<2> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x1x16x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x16x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps_one) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps_two) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x?xf32>
  %hi_cast = "onnx.Cast"(%hi) {saturate = 1 : si64, to = f32} : (tensor<1x3x16x?xf32>) -> tensor<1x3x16x4xf32>
  %neg = "onnx.Neg"(%hi_cast) : (tensor<1x3x16x4xf32>) -> tensor<1x3x16x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x16x4xf32>, tensor<1x3x16x4xf32>) -> tensor<1x3x16x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %m_data = "onnx.Mul"(%patches, %cos) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x16x8xf32>, tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32>
  return %out : tensor<1x3x16x8xf32>

// CHECK-LABEL: func.func @rope_negative_slice_step
// CHECK-NOT:   onnx.RotaryEmbedding
}

// -----

// Negative: the second Mul also has a constant data input

func.func @rope_negative_two_const_mul(%patches: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %starts_lo = onnx.Constant dense<0> : tensor<1xi64>
  %ends_lo   = onnx.Constant dense<4> : tensor<1xi64>
  %starts_hi = onnx.Constant dense<4> : tensor<1xi64>
  %ends_hi   = onnx.Constant dense<8> : tensor<1xi64>
  %axes      = onnx.Constant dense<3> : tensor<1xi64>
  %steps     = onnx.Constant dense<1> : tensor<1xi64>
  %cos = onnx.Constant dense<2.0> : tensor<1x1x16x8xf32>
  %sin = onnx.Constant dense<3.0> : tensor<1x1x16x8xf32>
  %const_data = onnx.Constant dense<7.0> : tensor<1x3x16x8xf32>
  %lo = "onnx.Slice"(%patches, %starts_lo, %ends_lo, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %hi = "onnx.Slice"(%patches, %starts_hi, %ends_hi, %axes, %steps) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x16x4xf32>
  %neg = "onnx.Neg"(%hi) : (tensor<1x3x16x4xf32>) -> tensor<1x3x16x4xf32>
  %rot = "onnx.Concat"(%neg, %lo) {axis = -1 : si64} : (tensor<1x3x16x4xf32>, tensor<1x3x16x4xf32>) -> tensor<1x3x16x8xf32>
  %m_rot = "onnx.Mul"(%rot, %sin) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %m_data = "onnx.Mul"(%const_data, %cos) : (tensor<1x3x16x8xf32>, tensor<1x1x16x8xf32>) -> tensor<1x3x16x8xf32>
  %out = "onnx.Add"(%m_data, %m_rot) : (tensor<1x3x16x8xf32>, tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32>
  return %out : tensor<1x3x16x8xf32>

// CHECK-LABEL: func.func @rope_negative_two_const_mul
// CHECK-NOT:   onnx.RotaryEmbedding
}
