// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt -onnx-hybrid-transform="enable-rotary-embedding-recompose=true" %s | FileCheck %s

// End-to-end check that the three RoPE-related patterns compose under the
// hybrid transform pass:
//
//   RecomposeRotaryEmbeddingPattern
//     -> rewrites the HF-style elementwise rotate_half subgraph into a
//        single onnx.RotaryEmbedding op operating on the patch tensor.
//
//   EliminateCarveOutAroundRotaryEmbeddingPattern
//     -> absorbs the surrounding `Concat(prefix, RoPE(patches))` into
//        the cos/sin caches by padding row 0 with the RoPE identity
//        (cos = 1, sin = 0). Constant propagation then folds the padded
//        Concat ops into dense constants.
//
//   FuseScaleIntoRotaryEmbeddingPattern
//     -> absorbs the trailing single-element scalar `onnx.Mul` into the
//        cos/sin caches. Constant propagation again folds the resulting
//        `Mul(constCos, constScale)` ops to dense constants.
//


func.func @rope_full_pipeline(%X: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
  %s0    = onnx.Constant dense<0> : tensor<1xi64>
  %s1    = onnx.Constant dense<1> : tensor<1xi64>
  %s4    = onnx.Constant dense<4> : tensor<1xi64>
  %s8    = onnx.Constant dense<8> : tensor<1xi64>
  %s16   = onnx.Constant dense<16> : tensor<1xi64>
  %ax2   = onnx.Constant dense<2> : tensor<1xi64>
  %ax3   = onnx.Constant dense<3> : tensor<1xi64>
  %step  = onnx.Constant dense<1> : tensor<1xi64>

  %cos   = onnx.Constant dense<2.000000e+00> : tensor<1x1x15x8xf32>
  %sin   = onnx.Constant dense<3.000000e+00> : tensor<1x1x15x8xf32>
  %scale = onnx.Constant dense<5.000000e+00> : tensor<1xf32>
  %none  = "onnx.NoValue"() {value} : () -> none

  %pre   = "onnx.Slice"(%X, %s0, %s1,  %ax2, %step) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x1x8xf32>
  %pat   = "onnx.Slice"(%X, %s1, %s16, %ax2, %step) : (tensor<1x3x16x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x8xf32>

  %lo    = "onnx.Slice"(%pat, %s0, %s4, %ax3, %step) : (tensor<1x3x15x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x4xf32>
  %hi    = "onnx.Slice"(%pat, %s4, %s8, %ax3, %step) : (tensor<1x3x15x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3x15x4xf32>
  %nhi   = "onnx.Neg"(%hi) : (tensor<1x3x15x4xf32>) -> tensor<1x3x15x4xf32>
  %rot   = "onnx.Concat"(%nhi, %lo) {axis = 3 : si64} : (tensor<1x3x15x4xf32>, tensor<1x3x15x4xf32>) -> tensor<1x3x15x8xf32>

  %mDat  = "onnx.Mul"(%pat, %cos) : (tensor<1x3x15x8xf32>, tensor<1x1x15x8xf32>) -> tensor<1x3x15x8xf32>
  %mRot  = "onnx.Mul"(%rot, %sin) : (tensor<1x3x15x8xf32>, tensor<1x1x15x8xf32>) -> tensor<1x3x15x8xf32>
  %rope  = "onnx.Add"(%mDat, %mRot) : (tensor<1x3x15x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x15x8xf32>

  %y     = "onnx.Concat"(%pre, %rope) {axis = 2 : si64} : (tensor<1x3x1x8xf32>, tensor<1x3x15x8xf32>) -> tensor<1x3x16x8xf32>
  %z     = "onnx.Mul"(%y, %scale) : (tensor<1x3x16x8xf32>, tensor<1xf32>) -> tensor<1x3x16x8xf32>
  return %z : tensor<1x3x16x8xf32>

// CHECK-LABEL:  func.func @rope_full_pipeline
// CHECK-SAME:   ([[X_:%.+]]: tensor<1x3x16x8xf32>) -> tensor<1x3x16x8xf32> {
// CHECK-DAG:       [[NONE_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[COS_:%.+]] = onnx.Constant {{.*}} : tensor<1x16x4xf32>
// CHECK-DAG:       [[SIN_:%.+]] = onnx.Constant {{.*}} : tensor<1x16x4xf32>
// CHECK:           [[OUT_:%.+]] = "onnx.RotaryEmbedding"([[X_]], [[COS_]], [[SIN_]], [[NONE_]]) {interleaved = 0 : si64, num_heads = 3 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x3x16x8xf32>, tensor<1x16x4xf32>, tensor<1x16x4xf32>, none) -> tensor<1x3x16x8xf32>
// CHECK-NEXT:      return [[OUT_]] : tensor<1x3x16x8xf32>
// CHECK-NOT:       onnx.Slice
// CHECK-NOT:       onnx.Neg
// CHECK-NOT:       onnx.Add
// CHECK-NOT:       onnx.Concat
// CHECK-NOT:       onnx.Mul
}
