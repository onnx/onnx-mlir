// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// Tests for extended layout transform patterns using the fused-op approach that
// are NOT covered by zhigh-fold-op-stick-unstick-fused-op.mlir:
//   v2: full LT→Reshape(split)→Transpose→Reshape(merge)→LT with factor=128
//   test_lt_reshape_lt: LT→Reshape(merge/squeeze)→LT, no transpose, no split
//
// Compare with zhigh-fold-op-stick-unstick.mlir (--disable-fused-op) for the
// non-fused zhigh.ExtendedLayoutTransform equivalents.

// -----

// layout reshape(split,factor=128) transpose reshape(merge) layout
// Differs from v1 in split factor (128 vs 64) and tensor widths (4096 vs 2048).

func.func @pattern_extended_layout_transform_v2(%arg0: tensor<3x?x4096xf32>, %arg1: tensor<4096x4096xf32>, %arg2: tensor<4096x512xf32>, %arg3: tensor<4096x512xf32>, %arg4: tensor<128x128xf32>) -> tensor<3x32x?x128xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = onnx.Constant dense<32> : tensor<1xi64>
    %2 = onnx.Constant dense<96> : tensor<1xi64>
    %3 = onnx.Constant dense<3> : tensor<1xi64>
    %4 = onnx.Constant dense<-1> : tensor<1xi64>
    %5 = onnx.Constant dense<128> : tensor<1xi64>
    %6 = "onnx.Dim"(%arg0) {axis = 1 : si64, onnx_node_name = "onnx.Dim_0"} : (tensor<3x?x4096xf32>) -> tensor<1xi64>
    %7 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<3x?x4096xf32>) -> tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %8 = "zhigh.Stick"(%arg1) {layout = "2D"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %9 = "zhigh.MatMul"(%7, %8, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4096x4096xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %10 = "onnx.LayoutTransform"(%9) : (tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x?x4096xf16>
    %11 = "onnx.Concat"(%3, %6, %4, %5) {axis = 0 : si64, onnx_node_name = "/model/layers.0/self_attn/Concat"} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %12 = "onnx.Reshape"(%10, %11) {allowzero = 0 : si64} : (tensor<3x?x4096xf16>, tensor<4xi64>) -> tensor<3x?x32x128xf16>
    %13 = "onnx.Transpose"(%12) {perm = [0, 2, 1, 3]} : (tensor<3x?x32x128xf16>) -> tensor<3x32x?x128xf16>
    %14 = "onnx.Concat"(%2, %6, %5) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %15 = "onnx.Reshape"(%13, %14) {allowzero = 0 : si64} : (tensor<3x32x?x128xf16>, tensor<3xi64>) -> tensor<96x?x128xf16>
    %16 = "onnx.LayoutTransform"(%15) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>} : (tensor<96x?x128xf16>) -> tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %17 = "zhigh.Stick"(%arg4) {layout = "2D"} : (tensor<128x128xf32>) -> tensor<128x128xf16, #zhigh.layout<{dataLayout = "2D"}>>
    %18 = "zhigh.MatMul"(%16, %17, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<128x128xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %19 = "zhigh.Unstick"(%18) : (tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x128xf32>
    %20 = "onnx.Concat"(%3, %1, %6, %5) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %21 = "onnx.Reshape"(%19, %20) {allowzero = 0 : si64} : (tensor<96x?x128xf32>, tensor<4xi64>) -> tensor<3x32x?x128xf32>
    return %21 : tensor<3x32x?x128xf32>

// CHECK-LABEL:  func.func @pattern_extended_layout_transform_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x?x4096xf32>, [[PARAM_1_:%.+]]: tensor<4096x4096xf32>, [[PARAM_2_:%.+]]: tensor<4096x512xf32>, [[PARAM_3_:%.+]]: tensor<4096x512xf32>, [[PARAM_4_:%.+]]: tensor<128x128xf32>) -> tensor<3x32x?x128xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<128> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Dim"([[PARAM_0_]]) <{axis = 1 : si64}> {onnx_node_name = "onnx.Dim_0"} : (tensor<3x?x4096xf32>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) <{layout = "3DS"}> : (tensor<3x?x4096xf32>) -> tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_6_:%.+]] = "zhigh.Stick"([[PARAM_1_]]) <{layout = "2D"}> : (tensor<4096x4096xf32>) -> tensor<4096x4096xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.MatMul"([[VAR_5_]], [[VAR_6_]], [[VAR_0_]]) <{transposeA = 0 : si64, transposeB = 0 : si64}> : (tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4096x4096xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<3x?x4096xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Fused"([[VAR_7_]], {{.*}}) <{kind = "zhigh.extended_layout_transform"}>
// Verify the chain ops are inside the fused body:
// CHECK:           "onnx.LayoutTransform"
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<3x?x32x128xf16>
// CHECK:           "onnx.Transpose"
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<96x?x128xf16>
// CHECK:           "onnx.LayoutTransform"{{.*}}-> tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           onnx.Yield
// Verify stored params (split factor 128, merge at axis 0):
// CHECK:           reshapeSplitAxis = 2{{.*}}reshapeSplitFactor = 128
// Verify outer structure after the fused op:
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Stick"([[PARAM_4_]]) <{layout = "2D"}> : (tensor<128x128xf32>) -> tensor<128x128xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           [[VAR_10_:%.+]] = "zhigh.MatMul"([[VAR_8_]], [[VAR_9_]], [[VAR_0_]]) <{transposeA = 0 : si64, transposeB = 0 : si64}> : (tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<128x128xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:       [[VAR_11_:%.+]] = "zhigh.Unstick"([[VAR_10_]]) : (tensor<96x?x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<96x?x128xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_1_]], [[VAR_4_]], [[VAR_3_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Reshape"([[VAR_11_]], [[VAR_12_]]) <{allowzero = 0 : si64}> : (tensor<96x?x128xf32>, tensor<4xi64>) -> tensor<3x32x?x128xf32>
// CHECK:           return [[VAR_13_]] : tensor<3x32x?x128xf32>
// CHECK-NOT:       "zhigh.ExtendedLayoutTransform"
// CHECK:           }
}

// -----

// layout reshape(merge/squeeze) layout — no split, no transpose.
// The constant reshape target shape is cloned inside the fused body;
// the function argument is the only FusedOp input.

func.func @test_lt_reshape_lt(%arg0: tensor<1x12x64x256xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<12x64x256xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
    %0 = onnx.Constant dense<[12, 64, 256]> : tensor<3xi64>
    %1 = "onnx.LayoutTransform"(%arg0) : (tensor<1x12x64x256xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<1x12x64x256xf16>
    %2 = "onnx.Reshape"(%1, %0) <{allowzero = 0 : si64}> : (tensor<1x12x64x256xf16>, tensor<3xi64>) -> tensor<12x64x256xf16>
    %3 = "onnx.LayoutTransform"(%2) <{target_layout = #zhigh.layout<{dataLayout = "3DS"}>}> : (tensor<12x64x256xf16>) -> tensor<12x64x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    return %3 : tensor<12x64x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @test_lt_reshape_lt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x12x64x256xf16, #zhigh.layout<{dataLayout = "4D"}>>) -> tensor<12x64x256xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]]) <{kind = "zhigh.extended_layout_transform"}>
// Constant reshape target is cloned inside the fused body:
// CHECK:           onnx.Constant dense<[12, 64, 256]> : tensor<3xi64>
// Verify the chain ops are inside the fused body:
// CHECK:           "onnx.LayoutTransform"{{.*}}-> tensor<1x12x64x256xf16>
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<12x64x256xf16>
// CHECK:           "onnx.LayoutTransform"{{.*}}-> tensor<12x64x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           onnx.Yield
// Verify stored params (merge only, no split, no transpose):
// CHECK:           reshapeMergeAxis = 0{{.*}}reshapeSplitAxis = -1{{.*}}reshapeSplitFactor = 1
// CHECK:           return [[VAR_0_]] : tensor<12x64x256xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-NOT:       "zhigh.ExtendedLayoutTransform"
// CHECK:           }
}

// -----

// Fusion must NOT fire when an intermediate chain value has more than one use.
// Here the split-reshape result (%rs1) is returned directly in addition to
// being consumed by the transpose.  singleUserOfType<ONNXTransposeOp> returns
// null for %rs1 (two users), so steps 3-5 of detectIfBeneficial find nothing,
// the beneficial check fails, and the chain stays un-fused.

func.func @no_fuse_reshape_has_two_users(
    %arg0: tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>,
    %arg1: tensor<4xi64>, %arg2: tensor<3xi64>)
    -> (tensor<3x?x32x64xf16>, tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) {
  %lt  = "onnx.LayoutTransform"(%arg0) : (tensor<3x?x2048xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x?x2048xf16>
  %rs1 = "onnx.Reshape"(%lt, %arg1) <{allowzero = 0 : si64}> : (tensor<3x?x2048xf16>, tensor<4xi64>) -> tensor<3x?x32x64xf16>
  %tr  = "onnx.Transpose"(%rs1) {perm = [0, 2, 1, 3]} : (tensor<3x?x32x64xf16>) -> tensor<3x32x?x64xf16>
  %rs2 = "onnx.Reshape"(%tr, %arg2) <{allowzero = 0 : si64}> : (tensor<3x32x?x64xf16>, tensor<3xi64>) -> tensor<96x?x64xf16>
  %lt2 = "onnx.LayoutTransform"(%rs2) <{target_layout = #zhigh.layout<{dataLayout = "3DS"}>}> : (tensor<96x?x64xf16>) -> tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  // %rs1 is consumed by both %tr (chain) and the return — two users.
  return %rs1, %lt2 : tensor<3x?x32x64xf16>, tensor<96x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @no_fuse_reshape_has_two_users
// No onnx.Fused must be created — the chain ops remain in-place.
// CHECK-NOT:    "onnx.Fused"
// CHECK-NOT:    "zhigh.ExtendedLayoutTransform"
// CHECK:        "onnx.LayoutTransform"
// CHECK:        "onnx.Reshape"
// CHECK:        "onnx.Transpose"
// CHECK:        "onnx.Reshape"
// CHECK:        "onnx.LayoutTransform"
// CHECK:        return
}
