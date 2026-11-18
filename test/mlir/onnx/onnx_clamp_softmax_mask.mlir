// RUN: onnx-mlir-opt --clamp-softmax-mask %s -split-input-file | FileCheck %s

// COM: Grounded in file mode: onnx_clamp_softmax_mask.baseline.mlir holds the
// COM: expected output of the pass (same functions, clamped constants), and
// COM: `GroundLitTest.py test/mlir/onnx/onnx_clamp_softmax_mask.mlir` checks
// COM: that both compute the same results on CPU. Cases where the pass must
// COM: not change anything are in onnx_clamp_softmax_mask_keep.mlir.
// COM: Masked entries get weight 0 either way and fully masked rows are
// COM: uniform either way, so the results match to rounding.
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
// CHECK:           [[VAR_5_:%.+]] = "onnx.Sub"([[VAR_1_]], [[VAR_4_]]) : (tensor<f32>, tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Cast"([[VAR_5_]]) <{saturate = 1 : si64, to = i1}> : (tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xi1>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Where"([[VAR_6_]], [[VAR_0_]], [[VAR_5_]]) : (tensor<2x1x1x4xi1>, tensor<f32>, tensor<2x1x1x4xf32>) -> tensor<2x1x1x4xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Expand"([[VAR_7_]], [[VAR_3_]]) : (tensor<2x1x1x4xf32>, tensor<4xi64>) -> tensor<2x1x4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Where"([[PARAM_1_]], [[VAR_8_]], [[VAR_0_]]) : (tensor<1x1x4x4xi1>, tensor<2x1x4x4xf32>, tensor<f32>) -> tensor<2x1x4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Add"([[PARAM_2_]], [[VAR_8_]]) : (tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Softmax"([[VAR_10_]]) <{axis = -1 : si64}> : (tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Add"([[PARAM_3_]], [[VAR_9_]]) : (tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Softmax"([[VAR_12_]]) <{axis = -1 : si64}> : (tensor<2x1x4x4xf32>) -> tensor<2x1x4x4xf32>
// CHECK:           return [[VAR_11_]], [[VAR_13_]] : tensor<2x1x4x4xf32>, tensor<2x1x4x4xf32>
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
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Softmax"([[VAR_10_]]) <{axis = -1 : si64}> : (tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Add"([[PARAM_3_]], [[VAR_9_]]) : (tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Softmax"([[VAR_12_]]) <{axis = -1 : si64}> : (tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
// CHECK:           return [[VAR_11_]], [[VAR_13_]] : tensor<?x1x?x?xf32>, tensor<?x1x?x?xf32>
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
// CHECK:           [[VAR_6_:%.+]] = "onnx.Softmax"([[VAR_5_]]) <{axis = -1 : si64}> : (tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x1x3x3xf32>
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
// CHECK:           [[VAR_3_:%.+]] = "onnx.Where"([[PARAM_3_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_present_key_:%.+]], [[VAR_present_value_:%.+]], [[VAR_qk_matmul_output_:%.+]] = "onnx.Attention"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[VAR_3_]], [[VAR_0_]], [[VAR_0_]], [[VAR_0_]]) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}> : (tensor<1x1x2x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x1x3xf32>, none, none, none) -> (tensor<1x1x2x2xf32>, none, none, none)
// CHECK:           return [[VAR_Y_]] : tensor<1x1x2x2xf32>
// CHECK:         }

}

