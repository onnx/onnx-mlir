// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --rewrite-onnx-for-zhigh %s -split-input-file | FileCheck %s

// COM: Mask constant clamping in --rewrite-onnx-for-zhigh: a constant below
// COM: -1e9 (-1e4 for f16) that reaches a Softmax input or a float
// COM: onnx.Attention mask is replaced, at those uses, by a copy raised to
// COM: that value. The rest of the pass also applies (e.g. onnx.Attention is
// COM: decomposed). Cases where nothing must change are in
// COM: rewrite-onnx-for-zhigh-clamp-mask-keep.mlir.
// COM:
// COM: Grounded in file mode: rewrite-onnx-for-zhigh-clamp-mask.baseline.mlir
// COM: holds each function with the clamping applied by hand (and nothing
// COM: else), and `GroundLitTest.py <this file>` checks on CPU that both
// COM: compute the same results. Masked entries get weight 0 either way and
// COM: fully masked rows are uniform either way, so they match to rounding.
// GROUND-ALL: --rtol=1e-5 --atol=1e-6

// -----

// COM: Global padding mask and sliding-window mask sharing one -FLT_MAX fill
// COM: constant, each added to scores before a Softmax (as in ModernBERT /
// COM: granite-embedding-r2 exports). The fill becomes -1e9. The mask is ones
// COM: then zeros per row, so rows are partly or fully masked.
// GROUND-THIS: --input-value=0:soz-1
func.func @test_where_masks(%mask: tensor<2x4xf32> {onnx.name = "mask"}, %win: tensor<1x1x4x4xi1> {onnx.name = "win"}, %s0: tensor<2x1x4x4xf32> {onnx.name = "s0"}, %s1: tensor<2x1x4x4xf32> {onnx.name = "s1"}) -> (tensor<2x1x4x4xf32> {onnx.name = "p0"}, tensor<2x1x4x4xf32> {onnx.name = "p1"}) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %axes = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %shape = onnx.Constant dense<[2, 1, 4, 4]> : tensor<4xi64>
  %m4 = "onnx.Unsqueeze"(%mask, %axes) : (tensor<2x4xf32>, tensor<2xi64>) -> tensor<2x1x1x4xf32>
  %inv = "onnx.Sub"(%one, %m4) : (tensor<f32>, tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xf32>
  %pad = "onnx.Cast"(%inv) {saturate = 1 : si64, to = i1} : (tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xi1>
  %g = "onnx.Where"(%pad, %min, %inv) : (tensor<2x1x1x4xi1>, tensor<f32>, tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xf32>
  %ge = "onnx.Expand"(%g, %shape) : (tensor<2x1x1x4xf32>, tensor<4xi64>) -> tensor<2x1x4x4xf32>
  %w = "onnx.Where"(%win, %ge, %min) : (tensor<1x1x4x4xi1>, tensor<2x1x4x4xf32>, tensor<f32>) -> tensor<2x1x4x4xf32>
  %a0 = "onnx.Add"(%s0, %ge) : (tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
  %p0 = "onnx.Softmax"(%a0) {axis = -1 : si64} : (tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
  %a1 = "onnx.Add"(%s1, %w) : (tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
  %p1 = "onnx.Softmax"(%a1) {axis = -1 : si64} : (tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
  return %p0, %p1 : tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>


// CHECK-LABEL:  func.func @test_where_masks
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32> {onnx.name = "mask"}, [[PARAM_1_:%.+]]: tensor<1x1x4x4xi1> {onnx.name = "win"}, [[PARAM_2_:%.+]]: tensor<2x1x4x4xf32> {onnx.name = "s0"}, [[PARAM_3_:%.+]]: tensor<2x1x4x4xf32> {onnx.name = "s1"}) -> (tensor<2x1x4x4xf32> {onnx.name = "p0"}, tensor<2x1x4x4xf32> {onnx.name = "p1"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[2, 1, 4, 4]> : tensor<4xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_2_]]) : (tensor<2x4xf32>, tensor<2xi64>) -> tensor<2x1x1x4xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Shape"([[VAR_4_]]) <{start = 0 : si64}> : (tensor<2x1x1x4xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Expand"([[VAR_1_]], [[VAR_5_]]) : (tensor<f32>, tensor<4xi64>) -> tensor<2x1x1x4xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Sub"([[VAR_6_]], [[VAR_4_]]) : (tensor<2x1x1x4xf32>, tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Cast"([[VAR_7_]]) <{saturate = 1 : si64, to = i1}> : (tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xi1>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Where"([[VAR_8_]], [[VAR_0_]], [[VAR_7_]]) : (tensor<2x1x1x4xi1>, tensor<f32>, tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Expand"([[VAR_9_]], [[VAR_3_]]) : (tensor<2x1x1x4xf32>, tensor<4xi64>) -> tensor<2x1x4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Where"([[PARAM_1_]], [[VAR_10_]], [[VAR_0_]]) : (tensor<1x1x4x4xi1>, tensor<2x1x4x4xf32>, tensor<f32>) -> tensor<2x1x4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Add"([[PARAM_2_]], [[VAR_10_]]) : (tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Shape"([[VAR_12_]]) <{start = 0 : si64}> : (tensor<2x1x4x4xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_16_]], [[VAR_17_]], [[VAR_15_]], [[VAR_18_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           [[VAR_21_:%.+]] = "onnx.Concat"([[VAR_20_]], [[VAR_19_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_22_:%.+]] = "onnx.Reshape"([[VAR_12_]], [[VAR_21_]]) <{allowzero = 0 : si64}> : (tensor<2x1x4x4xf32>, tensor<3xi64>) -> tensor<2x4x4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Softmax"([[VAR_22_]]) <{axis = -1 : si64}> : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Shape"([[VAR_12_]]) <{start = 0 : si64}> : (tensor<2x1x4x4xf32>) -> tensor<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Reshape"([[VAR_23_]], [[VAR_24_]]) <{allowzero = 0 : si64}> : (tensor<2x4x4xf32>, tensor<4xi64>) -> tensor<2x1x4x4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Add"([[PARAM_3_]], [[VAR_11_]]) : (tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Shape"([[VAR_26_]]) <{start = 0 : si64}> : (tensor<2x1x4x4xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_28_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_32_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Slice"([[VAR_27_]], [[VAR_30_]], [[VAR_31_]], [[VAR_29_]], [[VAR_32_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_34_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Concat"([[VAR_34_]], [[VAR_33_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Reshape"([[VAR_26_]], [[VAR_35_]]) <{allowzero = 0 : si64}> : (tensor<2x1x4x4xf32>, tensor<3xi64>) -> tensor<2x4x4xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.Softmax"([[VAR_36_]]) <{axis = -1 : si64}> : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Shape"([[VAR_26_]]) <{start = 0 : si64}> : (tensor<2x1x4x4xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_39_:%.+]] = "onnx.Reshape"([[VAR_37_]], [[VAR_38_]]) <{allowzero = 0 : si64}> : (tensor<2x4x4xf32>, tensor<4xi64>) -> tensor<2x1x4x4xf32>
// CHECK:           return [[VAR_25_]], [[VAR_39_]] : tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>
// CHECK:         }

}

// -----

// COM: Same masks with all dimensions dynamic (batch and sequence), as in the
// COM: real model; the expanded shape comes from the scores.
// GROUND-THIS: --shape-info=0:2x5,1:1x1x5x5,2:2x1x5x5,3:2x1x5x5 --input-value=0:soz-1
func.func @test_where_masks_dyn(%mask: tensor<?x?xf32> {onnx.name = "mask"}, %win: tensor<1x1x?x?xi1> {onnx.name = "win"}, %s0: tensor<?x1x?x?xf32> {onnx.name = "s0"}, %s1: tensor<?x1x?x?xf32> {onnx.name = "s1"}) -> (tensor<?x1x?x?xf32> {onnx.name = "p0"}, tensor<?x1x?x?xf32> {onnx.name = "p1"}) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %axes = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %shape = "onnx.Shape"(%s0) : (tensor<?x1x?x?xf32>) -> tensor<4xi64>
  %m4 = "onnx.Unsqueeze"(%mask, %axes) : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x1x1x?xf32>
  %inv = "onnx.Sub"(%one, %m4) : (tensor<f32>, tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xf32>
  %pad = "onnx.Cast"(%inv) {saturate = 1 : si64, to = i1} : (tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xi1>
  %g = "onnx.Where"(%pad, %min, %inv) : (tensor<?x1x1x?xi1>, tensor<f32>, tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xf32>
  %ge = "onnx.Expand"(%g, %shape) : (tensor<?x1x1x?xf32>, tensor<4xi64>) -> tensor<?x1x?x?xf32>
  %w = "onnx.Where"(%win, %ge, %min) : (tensor<1x1x?x?xi1>, tensor<?x1x?x?xf32>, tensor<f32>) -> tensor<?x1x?x?xf32>
  %a0 = "onnx.Add"(%s0, %ge) : (tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  %p0 = "onnx.Softmax"(%a0) {axis = -1 : si64} : (tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  %a1 = "onnx.Add"(%s1, %w) : (tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  %p1 = "onnx.Softmax"(%a1) {axis = -1 : si64} : (tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  return %p0, %p1 : tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>


// CHECK-LABEL:  func.func @test_where_masks_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32> {onnx.name = "mask"}, [[PARAM_1_:%.+]]: tensor<1x1x?x?xi1> {onnx.name = "win"}, [[PARAM_2_:%.+]]: tensor<?x1x?x?xf32> {onnx.name = "s0"}, [[PARAM_3_:%.+]]: tensor<?x1x?x?xf32> {onnx.name = "s1"}) -> (tensor<?x1x?x?xf32> {onnx.name = "p0"}, tensor<?x1x?x?xf32> {onnx.name = "p1"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Shape"([[PARAM_2_]]) <{start = 0 : si64}> : (tensor<?x1x?x?xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_2_]]) : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x1x1x?xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Sub"([[VAR_1_]], [[VAR_4_]]) : (tensor<f32>, tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Cast"([[VAR_5_]]) <{saturate = 1 : si64, to = i1}> : (tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xi1>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Where"([[VAR_6_]], [[VAR_0_]], [[VAR_5_]]) : (tensor<?x1x1x?xi1>, tensor<f32>, tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Expand"([[VAR_7_]], [[VAR_3_]]) : (tensor<?x1x1x?xf32>, tensor<4xi64>) -> tensor<?x1x?x?xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Where"([[PARAM_1_]], [[VAR_8_]], [[VAR_0_]]) : (tensor<1x1x?x?xi1>, tensor<?x1x?x?xf32>, tensor<f32>) -> tensor<?x1x?x?xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Add"([[PARAM_2_]], [[VAR_8_]]) : (tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Shape"([[VAR_10_]]) <{start = 0 : si64}> : (tensor<?x1x?x?xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Slice"([[VAR_11_]], [[VAR_14_]], [[VAR_15_]], [[VAR_13_]], [[VAR_16_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Concat"([[VAR_12_]], [[VAR_17_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Reshape"([[VAR_10_]], [[VAR_18_]]) <{allowzero = 0 : si64}> : (tensor<?x1x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Softmax"([[VAR_19_]]) <{axis = -1 : si64}> : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Shape"([[VAR_10_]]) <{start = 0 : si64}> : (tensor<?x1x?x?xf32>) -> tensor<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Reshape"([[VAR_20_]], [[VAR_21_]]) <{allowzero = 0 : si64}> : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x1x?x?xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Add"([[PARAM_3_]], [[VAR_9_]]) : (tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Shape"([[VAR_23_]]) <{start = 0 : si64}> : (tensor<?x1x?x?xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_25_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_26_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_27_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_28_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_24_]], [[VAR_27_]], [[VAR_28_]], [[VAR_26_]], [[VAR_29_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_31_:%.+]] = "onnx.Concat"([[VAR_25_]], [[VAR_30_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_32_:%.+]] = "onnx.Reshape"([[VAR_23_]], [[VAR_31_]]) <{allowzero = 0 : si64}> : (tensor<?x1x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Softmax"([[VAR_32_]]) <{axis = -1 : si64}> : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Shape"([[VAR_23_]]) <{start = 0 : si64}> : (tensor<?x1x?x?xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Reshape"([[VAR_33_]], [[VAR_34_]]) <{allowzero = 0 : si64}> : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x1x?x?xf32>
// CHECK:           return [[VAR_22_]], [[VAR_35_]] : tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>
// CHECK:         }

}

// -----

// COM: Mixed static and dynamic dimensions: static batch, dynamic sequence.
// GROUND-THIS: --shape-info=0:2x7,1:2x3x7 --input-value=0:soz-1
func.func @test_where_mask_mixed(%mask: tensor<2x?xf32> {onnx.name = "mask"}, %s: tensor<2x3x?xf32> {onnx.name = "s"}) -> (tensor<2x3x?xf32> {onnx.name = "p"}) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %m3 = "onnx.Unsqueeze"(%mask, %axes) : (tensor<2x?xf32>, tensor<1xi64>) -> tensor<2x1x?xf32>
  %pad = "onnx.Cast"(%m3) {saturate = 1 : si64, to = i1} : (tensor<2x1x?xf32>) -> tensor<2x1x?xi1>
  %m = "onnx.Where"(%pad, %zero, %min) : (tensor<2x1x?xi1>, tensor<f32>, tensor<f32>) -> tensor<2x1x?xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<2x3x?xf32>, tensor<2x1x?xf32>) -> tensor<2x3x?xf32>
  %p = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
  return %p : tensor<2x3x?xf32>


// CHECK-LABEL:  func.func @test_where_mask_mixed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x?xf32> {onnx.name = "mask"}, [[PARAM_1_:%.+]]: tensor<2x3x?xf32> {onnx.name = "s"}) -> (tensor<2x3x?xf32> {onnx.name = "p"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_2_]]) : (tensor<2x?xf32>, tensor<1xi64>) -> tensor<2x1x?xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Cast"([[VAR_3_]]) <{saturate = 1 : si64, to = i1}> : (tensor<2x1x?xf32>) -> tensor<2x1x?xi1>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Where"([[VAR_4_]], [[VAR_1_]], [[VAR_0_]]) : (tensor<2x1x?xi1>, tensor<f32>, tensor<f32>) -> tensor<2x1x?xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_5_]]) : (tensor<2x3x?xf32>, tensor<2x1x?xf32>) -> tensor<2x3x?xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Softmax"([[VAR_6_]]) <{axis = -1 : si64}> : (tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
// CHECK:           return [[VAR_7_]] : tensor<2x3x?xf32>
// CHECK:         }

}

// -----

// COM: A constant causal mask (with -inf and -FLT_MAX) added to a padding
// COM: mask: both constants are clamped, element by element for the causal one.
func.func @test_added_masks(%pad: tensor<1x1x1x3xi1> {onnx.name = "pad"}, %s: tensor<1x1x3x3xf32> {onnx.name = "s"}) -> (tensor<1x1x3x3xf32> {onnx.name = "y"}) {
  %ninf = onnx.Constant dense<0xFF800000> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %causal = onnx.Constant dense<[[[[0.000000e+00, 0xFF800000, 0xFF800000], [0.000000e+00, 0.000000e+00, -3.40282347E+38], [0.000000e+00, 0.000000e+00, 0.000000e+00]]]]> : tensor<1x1x3x3xf32>
  %p = "onnx.Where"(%pad, %ninf, %zero) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
  %m = "onnx.Add"(%causal, %p) : (tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x3x3xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %y : tensor<1x1x3x3xf32>


// CHECK-LABEL:  func.func @test_added_masks
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x1x3xi1> {onnx.name = "pad"}, [[PARAM_1_:%.+]]: tensor<1x1x3x3xf32> {onnx.name = "s"}) -> (tensor<1x1x3x3xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<{{.}}[{{.}}[0.000000e+00, -1.000000e+09, -1.000000e+09], [0.000000e+00, 0.000000e+00, -1.000000e+09], [0.000000e+00, 0.000000e+00, 0.000000e+00]{{.}}{{.}}]> : tensor<1x1x3x3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Where"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[VAR_2_]], [[VAR_3_]]) : (tensor<1x1x3x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x3x3xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_4_]]) : (tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Shape"([[VAR_5_]]) <{start = 0 : si64}> : (tensor<1x1x3x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Slice"([[VAR_6_]], [[VAR_9_]], [[VAR_10_]], [[VAR_8_]], [[VAR_11_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Concat"([[VAR_13_]], [[VAR_12_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Reshape"([[VAR_5_]], [[VAR_14_]]) <{allowzero = 0 : si64}> : (tensor<1x1x3x3xf32>, tensor<3xi64>) -> tensor<1x3x3xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Softmax"([[VAR_15_]]) <{axis = -1 : si64}> : (tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Shape"([[VAR_5_]]) <{start = 0 : si64}> : (tensor<1x1x3x3xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_18_:%.+]] = "onnx.Reshape"([[VAR_16_]], [[VAR_17_]]) <{allowzero = 0 : si64}> : (tensor<1x3x3xf32>, tensor<4xi64>) -> tensor<1x1x3x3xf32>
// CHECK:           return [[VAR_18_]] : tensor<1x1x3x3xf32>
// CHECK:         }

}

// -----

// COM: f16 mask: the fill becomes -1e4 (-1e9 does not fit in f16).
func.func @test_f16(%pad: tensor<1x4xi1> {onnx.name = "pad"}, %s: tensor<3x4xf16> {onnx.name = "s"}) -> (tensor<3x4xf16> {onnx.name = "y"}) {
  %min = onnx.Constant dense<-6.550400e+04> : tensor<f16>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f16>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f16>, tensor<f16>) -> tensor<1x4xf16>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf16>, tensor<1x4xf16>) -> tensor<3x4xf16>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf16>) -> tensor<3x4xf16>
  return %y : tensor<3x4xf16>


// CHECK-LABEL:  func.func @test_f16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xi1> {onnx.name = "pad"}, [[PARAM_1_:%.+]]: tensor<3x4xf16> {onnx.name = "s"}) -> (tensor<3x4xf16> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1.000000e+04> : tensor<f16>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f16>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Where"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<1x4xi1>, tensor<f16>, tensor<f16>) -> tensor<1x4xf16>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_2_]]) : (tensor<3x4xf16>, tensor<1x4xf16>) -> tensor<3x4xf16>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Softmax"([[VAR_3_]]) <{axis = -1 : si64}> : (tensor<3x4xf16>) -> tensor<3x4xf16>
// CHECK:           return [[VAR_4_]] : tensor<3x4xf16>
// CHECK:         }

}

// -----

// COM: Float attn_mask of onnx.Attention is a sink as well.
func.func @test_attention_mask(%Q: tensor<1x1x2x2xf32> {onnx.name = "q"}, %K: tensor<1x1x3x2xf32> {onnx.name = "k"}, %V: tensor<1x1x3x2xf32> {onnx.name = "v"}, %pad: tensor<1x1x1x3xi1> {onnx.name = "pad"}) -> (tensor<1x1x2x2xf32> {onnx.name = "y"}) {
  %none = "onnx.NoValue"() {value} : () -> none
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
  %Y, %pk, %pv, %qk = "onnx.Attention"(%Q, %K, %V, %m, %none, %none, %none) {is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32} : (tensor<1x1x2x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x1x3xf32>, none, none, none) -> (tensor<1x1x2x2xf32>, none, none, none)
  return %Y : tensor<1x1x2x2xf32>


// CHECK-LABEL:  func.func @test_attention_mask
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x2xf32> {onnx.name = "q"}, [[PARAM_1_:%.+]]: tensor<1x1x3x2xf32> {onnx.name = "k"}, [[PARAM_2_:%.+]]: tensor<1x1x3x2xf32> {onnx.name = "v"}, [[PARAM_3_:%.+]]: tensor<1x1x1x3xi1> {onnx.name = "pad"}) -> (tensor<1x1x2x2xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Where"([[PARAM_3_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) <{perm = [0, 1, 3, 2]}> : (tensor<1x1x3x2xf32>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Shape"([[PARAM_0_]]) <{start = 0 : si64}> : (tensor<1x1x2x2xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Slice"([[VAR_5_]], [[VAR_8_]], [[VAR_9_]], [[VAR_7_]], [[VAR_10_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Concat"([[VAR_12_]], [[VAR_11_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_13_]]) <{allowzero = 0 : si64}> : (tensor<1x1x2x2xf32>, tensor<3xi64>) -> tensor<1x2x2xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Shape"([[VAR_4_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_18_]], [[VAR_19_]], [[VAR_17_]], [[VAR_20_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Concat"([[VAR_22_]], [[VAR_21_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Reshape"([[VAR_4_]], [[VAR_23_]]) <{allowzero = 0 : si64}> : (tensor<1x1x2x3xf32>, tensor<3xi64>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.MatMul"([[VAR_14_]], [[VAR_24_]]) : (tensor<1x2x2xf32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Shape"([[PARAM_0_]]) <{start = 0 : si64}> : (tensor<1x1x2x2xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Shape"([[VAR_4_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_28_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_32_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_33_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Slice"([[VAR_27_]], [[VAR_32_]], [[VAR_33_]], [[VAR_28_]], [[VAR_29_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.Slice"([[VAR_26_]], [[VAR_28_]], [[VAR_30_]], [[VAR_28_]], [[VAR_29_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Concat"([[VAR_35_]], [[VAR_34_]]) <{axis = 0 : si64}> : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.Reshape"([[VAR_25_]], [[VAR_36_]]) <{allowzero = 0 : si64}> : (tensor<1x2x3xf32>, tensor<4xi64>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = onnx.Constant dense<0.707106769> : tensor<1xf32>
// CHECK:           [[VAR_39_:%.+]] = "onnx.Shape"([[VAR_37_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_40_:%.+]] = "onnx.Expand"([[VAR_38_]], [[VAR_39_]]) : (tensor<1xf32>, tensor<4xi64>) -> tensor<1x1x2x3xf32>
// CHECK:           [[VAR_41_:%.+]] = "onnx.Mul"([[VAR_37_]], [[VAR_40_]]) : (tensor<1x1x2x3xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32>
// CHECK:           [[VAR_42_:%.+]] = "onnx.Add"([[VAR_41_]], [[VAR_3_]]) : (tensor<1x1x2x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = "onnx.Shape"([[VAR_42_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_44_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_45_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_46_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_48_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_49_:%.+]] = "onnx.Slice"([[VAR_43_]], [[VAR_46_]], [[VAR_47_]], [[VAR_45_]], [[VAR_48_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_51_:%.+]] = "onnx.Concat"([[VAR_50_]], [[VAR_49_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_52_:%.+]] = "onnx.Reshape"([[VAR_42_]], [[VAR_51_]]) <{allowzero = 0 : si64}> : (tensor<1x1x2x3xf32>, tensor<3xi64>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_53_:%.+]] = "onnx.Softmax"([[VAR_52_]]) <{axis = -1 : si64}> : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_54_:%.+]] = "onnx.Shape"([[VAR_42_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_55_:%.+]] = "onnx.Reshape"([[VAR_53_]], [[VAR_54_]]) <{allowzero = 0 : si64}> : (tensor<1x2x3xf32>, tensor<4xi64>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_56_:%.+]] = "onnx.Shape"([[VAR_55_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_57_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_58_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_59_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_60_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_61_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_62_:%.+]] = "onnx.Slice"([[VAR_56_]], [[VAR_59_]], [[VAR_60_]], [[VAR_58_]], [[VAR_61_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_63_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_64_:%.+]] = "onnx.Concat"([[VAR_63_]], [[VAR_62_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_65_:%.+]] = "onnx.Reshape"([[VAR_55_]], [[VAR_64_]]) <{allowzero = 0 : si64}> : (tensor<1x1x2x3xf32>, tensor<3xi64>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_66_:%.+]] = "onnx.Shape"([[PARAM_2_]]) <{start = 0 : si64}> : (tensor<1x1x3x2xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_67_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_68_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_69_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_70_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_71_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_72_:%.+]] = "onnx.Slice"([[VAR_66_]], [[VAR_69_]], [[VAR_70_]], [[VAR_68_]], [[VAR_71_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_73_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_74_:%.+]] = "onnx.Concat"([[VAR_73_]], [[VAR_72_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_75_:%.+]] = "onnx.Reshape"([[PARAM_2_]], [[VAR_74_]]) <{allowzero = 0 : si64}> : (tensor<1x1x3x2xf32>, tensor<3xi64>) -> tensor<1x3x2xf32>
// CHECK-DAG:       [[VAR_76_:%.+]] = "onnx.MatMul"([[VAR_65_]], [[VAR_75_]]) : (tensor<1x2x3xf32>, tensor<1x3x2xf32>) -> tensor<1x2x2xf32>
// CHECK-DAG:       [[VAR_77_:%.+]] = "onnx.Shape"([[VAR_55_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_78_:%.+]] = "onnx.Shape"([[PARAM_2_]]) <{start = 0 : si64}> : (tensor<1x1x3x2xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_79_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_80_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_81_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_82_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_83_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_84_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_85_:%.+]] = "onnx.Slice"([[VAR_78_]], [[VAR_83_]], [[VAR_84_]], [[VAR_79_]], [[VAR_80_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_86_:%.+]] = "onnx.Slice"([[VAR_77_]], [[VAR_79_]], [[VAR_81_]], [[VAR_79_]], [[VAR_80_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_87_:%.+]] = "onnx.Concat"([[VAR_86_]], [[VAR_85_]]) <{axis = 0 : si64}> : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_88_:%.+]] = "onnx.Reshape"([[VAR_76_]], [[VAR_87_]]) <{allowzero = 0 : si64}> : (tensor<1x2x2xf32>, tensor<4xi64>) -> tensor<1x1x2x2xf32>
// CHECK-DAG:       [[VAR_89_:%.+]] = "onnx.NoValue"() : () -> none
// CHECK:           return [[VAR_88_]] : tensor<1x1x2x2xf32>
// CHECK:         }

}

// -----

// COM: The -FLT_MAX constant is also used as a sentinel outside the mask
// COM: (Max with the scores, returned): the Where gets a clamped copy, and the
// COM: Max keeps the original constant.
func.func @test_shared_sentinel(%pad: tensor<1x4xi1> {onnx.name = "pad"}, %s: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}, tensor<3x4xf32> {onnx.name = "r"}) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %r = "onnx.Max"(%s, %min) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
  return %y, %r : tensor<3x4xf32>, tensor<3x4xf32>

// CHECK-LABEL:  func.func @test_shared_sentinel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xi1> {onnx.name = "pad"}, [[PARAM_1_:%.+]]: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}, tensor<3x4xf32> {onnx.name = "r"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Where"([[PARAM_0_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_3_]]) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Softmax"([[VAR_4_]]) <{axis = -1 : si64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Max"([[PARAM_1_]], [[VAR_0_]]) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_5_]], [[VAR_6_]] : tensor<3x4xf32>, tensor<3x4xf32>
// CHECK:         }

}

