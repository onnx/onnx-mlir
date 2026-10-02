// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// Tests for the zhigh.unstick-split-heads FusedOp pattern.
// The pass wraps the chain:
//   ZHighUnstickOp -> ONNXReshapeOp -> [ONNXTransposeOp] -> ONNXSplitOp
// into an onnx.Fused region with kind = "zhigh.unstick-split-heads" and one
// result per Split result. The Reshape splits the innermost dim C into
// (N, H, D); the Split cuts the N axis into N slices of size 1. The Squeeze
// ops after the Split stay outside the fused op.

// -----

// The attention QKV pattern of granite-embedding-r2: dynamic batch and
// sequence, C = 1152 = 3 x 12 x 32, transpose [0, 3, 2, 1, 4], so the N axis
// lands at position 2. As in the model, the Reshape's batch dim comes from
// another input (the attention mask), related to the Unstick input through
// the onnx.dim_params names; the Reshape shape stays an external input.

func.func @unstick_split_heads_qkv(
    %arg0: tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>> {onnx.dim_params = "0:batch_size,1:sequence_length"},
    %arg1: tensor<?x?xi64> {onnx.dim_params = "0:batch_size,1:sequence_length"})
    -> (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) {
  %cm1  = onnx.Constant dense<-1> : tensor<1xi64>
  %c3   = onnx.Constant dense<3> : tensor<1xi64>
  %c12  = onnx.Constant dense<12> : tensor<1xi64>
  %c32  = onnx.Constant dense<32> : tensor<1xi64>
  %ones = onnx.Constant dense<1> : tensor<3xi64>
  %axes = onnx.Constant dense<2> : tensor<1xi64>
  %d0 = "onnx.Dim"(%arg1) <{axis = 0 : si64}> : (tensor<?x?xi64>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%d0, %cm1, %c3, %c12, %c32) <{axis = 0 : si64}>
         : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5xi64>
  %u = "zhigh.Unstick"(%arg0)
         : (tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x1152xf32>
  %r = "onnx.Reshape"(%u, %shape) <{allowzero = 0 : si64}>
         : (tensor<?x?x1152xf32>, tensor<5xi64>) -> tensor<?x?x3x12x32xf32>
  %t = "onnx.Transpose"(%r) <{perm = [0, 3, 2, 1, 4]}>
         : (tensor<?x?x3x12x32xf32>) -> tensor<?x12x3x?x32xf32>
  %s:3 = "onnx.Split"(%t, %ones) <{axis = 2 : si64}>
         : (tensor<?x12x3x?x32xf32>, tensor<3xi64>)
         -> (tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>)
  %q = "onnx.Squeeze"(%s#0, %axes) : (tensor<?x12x1x?x32xf32>, tensor<1xi64>) -> tensor<?x12x?x32xf32>
  %k = "onnx.Squeeze"(%s#1, %axes) : (tensor<?x12x1x?x32xf32>, tensor<1xi64>) -> tensor<?x12x?x32xf32>
  %v = "onnx.Squeeze"(%s#2, %axes) : (tensor<?x12x1x?x32xf32>, tensor<1xi64>) -> tensor<?x12x?x32xf32>
  return %q, %k, %v : tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>

// CHECK-LABEL:  func.func @unstick_split_heads_qkv
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>> {onnx.dim_params = "0:batch_size,1:sequence_length"}, [[PARAM_1_:%.+]]: tensor<?x?xi64> {onnx.dim_params = "0:batch_size,1:sequence_length"}) -> (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Dim"([[PARAM_1_]]) <{axis = 0 : si64}> : (tensor<?x?xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Concat"([[VAR_5_]], [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5xi64>
// CHECK:           [[VAR_7_:%.+]]:3 = "onnx.Fused"([[PARAM_0_]], [[VAR_6_]]) <{kind = "zhigh.unstick-split-heads"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>, [[REGION_ARG_1_:%.+]]: tensor<5xi64>):
// CHECK-DAG:         [[VAR_11_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:         [[VAR_12_:%.+]] = "zhigh.Unstick"([[REGION_ARG_0_]]) : (tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x1152xf32>
// CHECK:             [[VAR_13_:%.+]] = "onnx.Reshape"([[VAR_12_]], [[REGION_ARG_1_]]) <{allowzero = 0 : si64}> : (tensor<?x?x1152xf32>, tensor<5xi64>) -> tensor<?x?x3x12x32xf32>
// CHECK:             [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_13_]]) <{perm = [0, 3, 2, 1, 4]}> : (tensor<?x?x3x12x32xf32>) -> tensor<?x12x3x?x32xf32>
// CHECK:             [[VAR_15_:%.+]]:3 = "onnx.Split"([[VAR_14_]], [[VAR_11_]]) <{axis = 2 : si64}> : (tensor<?x12x3x?x32xf32>, tensor<3xi64>) -> (tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>)
// CHECK:             onnx.Yield [[VAR_15_]]#0, [[VAR_15_]]#1, [[VAR_15_]]#2 : tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>
// CHECK:           }) {headDim = 32 : i64, numHeads = 12 : i64, numSplits = 3 : i64, onnx_node_name = "zhigh.Unstick-onnx.Reshape-onnx.Transpose-onnx.Split", splitAxis = 2 : i64, transposePattern = [0, 3, 2, 1, 4]} : (tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<5xi64>) -> (tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Squeeze"([[VAR_7_]]#0, [[VAR_4_]]) : (tensor<?x12x1x?x32xf32>, tensor<1xi64>) -> tensor<?x12x?x32xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Squeeze"([[VAR_7_]]#1, [[VAR_4_]]) : (tensor<?x12x1x?x32xf32>, tensor<1xi64>) -> tensor<?x12x?x32xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Squeeze"([[VAR_7_]]#2, [[VAR_4_]]) : (tensor<?x12x1x?x32xf32>, tensor<1xi64>) -> tensor<?x12x?x32xf32>
// CHECK:           return [[VAR_8_]], [[VAR_9_]], [[VAR_10_]] : tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>
// CHECK:         }

}

// -----

// No transpose, static shapes, 3D layout: the N axis stays at position 2.

func.func @unstick_split_heads_no_transpose(%arg0: tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3D"}>>)
    -> (tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>) {
  %shape = onnx.Constant dense<[2, 8, 3, 2, 32]> : tensor<5xi64>
  %ones  = onnx.Constant dense<1> : tensor<3xi64>
  %u = "zhigh.Unstick"(%arg0)
         : (tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<2x8x192xf32>
  %r = "onnx.Reshape"(%u, %shape) <{allowzero = 0 : si64}>
         : (tensor<2x8x192xf32>, tensor<5xi64>) -> tensor<2x8x3x2x32xf32>
  %s:3 = "onnx.Split"(%r, %ones) <{axis = 2 : si64}>
         : (tensor<2x8x3x2x32xf32>, tensor<3xi64>)
         -> (tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>)
  return %s#0, %s#1, %s#2 : tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>

// CHECK-LABEL:  func.func @unstick_split_heads_no_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>) {
// CHECK:           [[VAR_0_:%.+]]:3 = "onnx.Fused"([[PARAM_0_]]) <{kind = "zhigh.unstick-split-heads"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3D"}>>):
// CHECK-DAG:         [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:         [[VAR_2_:%.+]] = onnx.Constant dense<[2, 8, 3, 2, 32]> : tensor<5xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = "zhigh.Unstick"([[REGION_ARG_0_]]) : (tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<2x8x192xf32>
// CHECK:             [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<2x8x192xf32>, tensor<5xi64>) -> tensor<2x8x3x2x32xf32>
// CHECK:             [[VAR_5_:%.+]]:3 = "onnx.Split"([[VAR_4_]], [[VAR_1_]]) <{axis = 2 : si64}> : (tensor<2x8x3x2x32xf32>, tensor<3xi64>) -> (tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>)
// CHECK:             onnx.Yield [[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2 : tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>
// CHECK:           }) {headDim = 32 : i64, numHeads = 2 : i64, numSplits = 3 : i64, onnx_node_name = "zhigh.Unstick-onnx.Reshape-onnx.Split", splitAxis = 2 : i64} : (tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>)
// CHECK:           return [[VAR_0_]]#0, [[VAR_0_]]#1, [[VAR_0_]]#2 : tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>
// CHECK:         }

}

// -----

// D = 64 (one head per stick), a different transpose that moves N to
// position 1, a negative split axis, and a Split with no split operand.

func.func @unstick_split_heads_d64(%arg0: tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>)
    -> (tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>) {
  %shape = onnx.Constant dense<[2, 8, 3, 2, 64]> : tensor<5xi64>
  %none  = "onnx.NoValue"() {value} : () -> none
  %u = "zhigh.Unstick"(%arg0)
         : (tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x384xf32>
  %r = "onnx.Reshape"(%u, %shape) <{allowzero = 0 : si64}>
         : (tensor<2x8x384xf32>, tensor<5xi64>) -> tensor<2x8x3x2x64xf32>
  %t = "onnx.Transpose"(%r) <{perm = [0, 2, 3, 1, 4]}>
         : (tensor<2x8x3x2x64xf32>) -> tensor<2x3x2x8x64xf32>
  %s:3 = "onnx.Split"(%t, %none) <{axis = -4 : si64, num_outputs = 3 : si64}>
         : (tensor<2x3x2x8x64xf32>, none)
         -> (tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>)
  return %s#0, %s#1, %s#2 : tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>

// CHECK-LABEL:  func.func @unstick_split_heads_d64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> (tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>) {
// CHECK:           [[VAR_0_:%.+]]:3 = "onnx.Fused"([[PARAM_0_]]) <{kind = "zhigh.unstick-split-heads"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>):
// CHECK-DAG:         [[VAR_1_:%.+]] = onnx.Constant dense<[2, 8, 3, 2, 64]> : tensor<5xi64>
// CHECK-DAG:         [[VAR_2_:%.+]] = "zhigh.Unstick"([[REGION_ARG_0_]]) : (tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x384xf32>
// CHECK:             [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<2x8x384xf32>, tensor<5xi64>) -> tensor<2x8x3x2x64xf32>
// CHECK-DAG:         [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) <{perm = [0, 2, 3, 1, 4]}> : (tensor<2x8x3x2x64xf32>) -> tensor<2x3x2x8x64xf32>
// CHECK-DAG:         [[VAR_5_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK:             [[VAR_6_:%.+]]:3 = "onnx.Split"([[VAR_4_]], [[VAR_5_]]) <{axis = -4 : si64, num_outputs = 3 : si64}> : (tensor<2x3x2x8x64xf32>, none) -> (tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>)
// CHECK:             onnx.Yield [[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2 : tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>
// CHECK:           }) {headDim = 64 : i64, numHeads = 2 : i64, numSplits = 3 : i64, onnx_node_name = "zhigh.Unstick-onnx.Reshape-onnx.Transpose-onnx.Split", splitAxis = 1 : i64, transposePattern = [0, 2, 3, 1, 4]} : (tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> (tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>)
// CHECK:           return [[VAR_0_]]#0, [[VAR_0_]]#1, [[VAR_0_]]#2 : tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>, tensor<2x1x2x8x64xf32>
// CHECK:         }

}

// -----

// Negative: the Reshape result has a second use, so it cannot move into the
// fused op body.

func.func @no_fuse_reshape_extra_use(%arg0: tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>)
    -> (tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x3x2x32xf32>) {
  %shape = onnx.Constant dense<[2, 8, 3, 2, 32]> : tensor<5xi64>
  %ones  = onnx.Constant dense<1> : tensor<3xi64>
  %u = "zhigh.Unstick"(%arg0)
         : (tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x192xf32>
  %r = "onnx.Reshape"(%u, %shape) <{allowzero = 0 : si64}>
         : (tensor<2x8x192xf32>, tensor<5xi64>) -> tensor<2x8x3x2x32xf32>
  %s:3 = "onnx.Split"(%r, %ones) <{axis = 2 : si64}>
         : (tensor<2x8x3x2x32xf32>, tensor<3xi64>)
         -> (tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>)
  return %s#0, %s#1, %s#2, %r : tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x3x2x32xf32>

// CHECK-LABEL:  func.func @no_fuse_reshape_extra_use
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> (tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x3x2x32xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[2, 8, 3, 2, 32]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x192xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x8x192xf32>, tensor<5xi64>) -> tensor<2x8x3x2x32xf32>
// CHECK:           [[VAR_4_:%.+]]:3 = "onnx.Split"([[VAR_3_]], [[VAR_1_]]) <{axis = 2 : si64}> : (tensor<2x8x3x2x32xf32>, tensor<3xi64>) -> (tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>)
// CHECK:           return [[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2, [[VAR_3_]] : tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x1x2x32xf32>, tensor<2x8x3x2x32xf32>
// CHECK:         }

}

// -----

// Negative: the Split does not cut N into unit slices.

func.func @no_fuse_split_not_unit(%arg0: tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>)
    -> (tensor<2x8x2x2x32xf32>, tensor<2x8x1x2x32xf32>) {
  %shape = onnx.Constant dense<[2, 8, 3, 2, 32]> : tensor<5xi64>
  %sizes = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %u = "zhigh.Unstick"(%arg0)
         : (tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x192xf32>
  %r = "onnx.Reshape"(%u, %shape) <{allowzero = 0 : si64}>
         : (tensor<2x8x192xf32>, tensor<5xi64>) -> tensor<2x8x3x2x32xf32>
  %s:2 = "onnx.Split"(%r, %sizes) <{axis = 2 : si64}>
         : (tensor<2x8x3x2x32xf32>, tensor<2xi64>)
         -> (tensor<2x8x2x2x32xf32>, tensor<2x8x1x2x32xf32>)
  return %s#0, %s#1 : tensor<2x8x2x2x32xf32>, tensor<2x8x1x2x32xf32>

// CHECK-LABEL:  func.func @no_fuse_split_not_unit
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> (tensor<2x8x2x2x32xf32>, tensor<2x8x1x2x32xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[2, 8, 3, 2, 32]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x192xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x8x192xf32>, tensor<5xi64>) -> tensor<2x8x3x2x32xf32>
// CHECK:           [[VAR_4_:%.+]]:2 = "onnx.Split"([[VAR_3_]], [[VAR_1_]]) <{axis = 2 : si64}> : (tensor<2x8x3x2x32xf32>, tensor<2xi64>) -> (tensor<2x8x2x2x32xf32>, tensor<2x8x1x2x32xf32>)
// CHECK:           return [[VAR_4_]]#0, [[VAR_4_]]#1 : tensor<2x8x2x2x32xf32>, tensor<2x8x1x2x32xf32>
// CHECK:         }

}

// -----

// Negative: D = 48 is neither 32 nor a multiple of 64.

func.func @no_fuse_head_dim_48(%arg0: tensor<2x8x576xf16, #zhigh.layout<{dataLayout = "3DS"}>>)
    -> (tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>) {
  %shape = onnx.Constant dense<[2, 8, 3, 4, 48]> : tensor<5xi64>
  %ones  = onnx.Constant dense<1> : tensor<3xi64>
  %u = "zhigh.Unstick"(%arg0)
         : (tensor<2x8x576xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x576xf32>
  %r = "onnx.Reshape"(%u, %shape) <{allowzero = 0 : si64}>
         : (tensor<2x8x576xf32>, tensor<5xi64>) -> tensor<2x8x3x4x48xf32>
  %s:3 = "onnx.Split"(%r, %ones) <{axis = 2 : si64}>
         : (tensor<2x8x3x4x48xf32>, tensor<3xi64>)
         -> (tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>)
  return %s#0, %s#1, %s#2 : tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>

// CHECK-LABEL:  func.func @no_fuse_head_dim_48
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x8x576xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> (tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[2, 8, 3, 4, 48]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<2x8x576xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x576xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x8x576xf32>, tensor<5xi64>) -> tensor<2x8x3x4x48xf32>
// CHECK:           [[VAR_4_:%.+]]:3 = "onnx.Split"([[VAR_3_]], [[VAR_1_]]) <{axis = 2 : si64}> : (tensor<2x8x3x4x48xf32>, tensor<3xi64>) -> (tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>)
// CHECK:           return [[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2 : tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>, tensor<2x8x1x4x48xf32>
// CHECK:         }

}

// -----

// Negative: the Transpose moves the innermost dim D.

func.func @no_fuse_transpose_moves_d(%arg0: tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>)
    -> (tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>) {
  %shape = onnx.Constant dense<[2, 8, 3, 2, 32]> : tensor<5xi64>
  %ones  = onnx.Constant dense<1> : tensor<3xi64>
  %u = "zhigh.Unstick"(%arg0)
         : (tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x192xf32>
  %r = "onnx.Reshape"(%u, %shape) <{allowzero = 0 : si64}>
         : (tensor<2x8x192xf32>, tensor<5xi64>) -> tensor<2x8x3x2x32xf32>
  %t = "onnx.Transpose"(%r) <{perm = [0, 1, 2, 4, 3]}>
         : (tensor<2x8x3x2x32xf32>) -> tensor<2x8x3x32x2xf32>
  %s:3 = "onnx.Split"(%t, %ones) <{axis = 2 : si64}>
         : (tensor<2x8x3x32x2xf32>, tensor<3xi64>)
         -> (tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>)
  return %s#0, %s#1, %s#2 : tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>

// CHECK-LABEL:  func.func @no_fuse_transpose_moves_d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> (tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[2, 8, 3, 2, 32]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<2x8x192xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x192xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x8x192xf32>, tensor<5xi64>) -> tensor<2x8x3x2x32xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_3_]]) <{perm = [0, 1, 2, 4, 3]}> : (tensor<2x8x3x2x32xf32>) -> tensor<2x8x3x32x2xf32>
// CHECK:           [[VAR_5_:%.+]]:3 = "onnx.Split"([[VAR_4_]], [[VAR_1_]]) <{axis = 2 : si64}> : (tensor<2x8x3x32x2xf32>, tensor<3xi64>) -> (tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>)
// CHECK:           return [[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2 : tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>, tensor<2x8x1x32x2xf32>
// CHECK:         }

}

// -----

// Negative: 4D layout (rank-4 unstick result).

func.func @no_fuse_4d_layout(%arg0: tensor<2x4x8x192xf16, #zhigh.layout<{dataLayout = "4D"}>>)
    -> (tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>) {
  %shape = onnx.Constant dense<[2, 4, 8, 3, 2, 32]> : tensor<6xi64>
  %ones  = onnx.Constant dense<1> : tensor<3xi64>
  %u = "zhigh.Unstick"(%arg0)
         : (tensor<2x4x8x192xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<2x4x8x192xf32>
  %r = "onnx.Reshape"(%u, %shape) <{allowzero = 0 : si64}>
         : (tensor<2x4x8x192xf32>, tensor<6xi64>) -> tensor<2x4x8x3x2x32xf32>
  %s:3 = "onnx.Split"(%r, %ones) <{axis = 3 : si64}>
         : (tensor<2x4x8x3x2x32xf32>, tensor<3xi64>)
         -> (tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>)
  return %s#0, %s#1, %s#2 : tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>
// CHECK-LABEL:  func.func @no_fuse_4d_layout
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8x192xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> (tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[2, 4, 8, 3, 2, 32]> : tensor<6xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "zhigh.Unstick"([[PARAM_0_]]) : (tensor<2x4x8x192xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<2x4x8x192xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) <{allowzero = 0 : si64}> : (tensor<2x4x8x192xf32>, tensor<6xi64>) -> tensor<2x4x8x3x2x32xf32>
// CHECK:           [[VAR_4_:%.+]]:3 = "onnx.Split"([[VAR_3_]], [[VAR_1_]]) <{axis = 3 : si64}> : (tensor<2x4x8x3x2x32xf32>, tensor<3xi64>) -> (tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>)
// CHECK:           return [[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2 : tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>, tensor<2x4x8x1x2x32xf32>
// CHECK:         }

}

