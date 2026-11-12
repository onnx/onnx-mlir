// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --canonicalize --fusion-op-stick-unstick %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --canonicalize --zhigh-decompose-stick-unstick --canonicalize --zhigh-recompose-to-stick-unstick --canonicalize --fusion-op-stick-unstick %s -split-input-file | FileCheck %s --check-prefix=PIPE

// Pins the interaction between the Expand-sinking canonicalization
// (SinkExpandBelowElementwiseOpPattern in ONNXOps/Canonicalize.cpp) and the
// zhigh.expand-mul-stick / zhigh.concat-expand-stick fusions.  The inputs are
// written in the order models produce, with the scalar Mul after the Expand:
//   [Concat ->] Unsqueeze -> Expand -> Mul(scalar) -> Reshape -> Stick
// The canonicalization rewrites Mul(Expand(x), k) into Expand(Mul(x, k)), and
// both fusions match the resulting Unsqueeze -> Mul -> Expand order, so each
// chain still becomes a single fused op with mulScalar = 2.0 (CHECK).
//
// The PIPE run adds the passes the NNPA pipeline (NNPACompilerUtils.cpp) runs
// between that canonicalization and the fusion: ZHighDecomposeStickUnstick
// rewrites Stick(Reshape(Expand(Mul(...)))) into Mul -> F32ToDLF16 -> Expand
// -> Reshape -> LayoutTransform, and ZHighRecomposeToStickUnstick does not
// undo it. The Concat-headed chain (granite-4's K path) still fuses, as a
// zhigh.concat-expand-stick with its 2-step (finalLayout) tail and
// mulScalar = 2.0. The standalone Unsqueeze chain does NOT fuse there:
// zhigh.expand-mul-stick only matches a tail ending in ZHighStickOp, and
// after the decomposition the chain ends in F32ToDLF16 ... LayoutTransform.

// -----

func.func @expand_mul_stick_after_canonicalize(%arg0: tensor<3x4x64xf32>)
    -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<1>             : tensor<1xi64>
  %shexp = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
  %scal  = onnx.Constant dense<2.000000e+00>  : tensor<f32>
  %shre  = onnx.Constant dense<[24, 4, 64]>   : tensor<3xi64>
  %unsq = "onnx.Unsqueeze"(%arg0, %axes)
            : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
  %mul  = "onnx.Mul"(%exp, %scal)
            : (tensor<3x8x4x64xf32>, tensor<f32>) -> tensor<3x8x4x64xf32>
  %resh = "onnx.Reshape"(%mul, %shre) <{allowzero = 0 : si64}>
            : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @expand_mul_stick_after_canonicalize
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]]) <{kind = "zhigh.expand-mul-stick"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<3x4x64xf32>):
// CHECK-DAG:         [[VAR_1_:%.+]] = onnx.Constant dense<[24, 4, 64]> : tensor<3xi64>
// CHECK-DAG:         [[VAR_2_:%.+]] = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:         [[VAR_4_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:             [[VAR_5_:%.+]] = "onnx.Unsqueeze"([[REGION_ARG_0_]], [[VAR_4_]]) : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
// CHECK:             [[VAR_6_:%.+]] = "onnx.Mul"([[VAR_5_]], [[VAR_3_]]) : (tensor<3x1x4x64xf32>, tensor<f32>) -> tensor<3x1x4x64xf32>
// CHECK:             [[VAR_7_:%.+]] = "onnx.Expand"([[VAR_6_]], [[VAR_2_]]) : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
// CHECK:             [[VAR_8_:%.+]] = "onnx.Reshape"([[VAR_7_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
// CHECK:             [[VAR_9_:%.+]] = "zhigh.Stick"([[VAR_8_]]) <{layout = "3DS"}> : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:             onnx.Yield [[VAR_9_]] : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }) {expansionN = 8 : i64, mulScalar = 2.000000e+00 : f32, onnx_node_name = "onnx.Unsqueeze-onnx.Mul-onnx.Expand-onnx.Reshape-zhigh.Stick", reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS", unsqueezedPosition = 1 : i64} : (tensor<3x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }

// PIPE-LABEL:  func.func @expand_mul_stick_after_canonicalize
// PIPE-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// PIPE-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// PIPE-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
// PIPE-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// PIPE-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[24, 4, 64]> : tensor<3xi64>
// PIPE:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
// PIPE:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_2_]]) : (tensor<3x1x4x64xf32>, tensor<f32>) -> tensor<3x1x4x64xf32>
// PIPE:           [[VAR_6_:%.+]] = "zhigh.F32ToDLF16"([[VAR_5_]]) : (tensor<3x1x4x64xf32>) -> tensor<3x1x4x64xf16>
// PIPE:           [[VAR_7_:%.+]] = "onnx.Expand"([[VAR_6_]], [[VAR_1_]]) : (tensor<3x1x4x64xf16>, tensor<4xi64>) -> tensor<3x8x4x64xf16>
// PIPE:           [[VAR_8_:%.+]] = "onnx.Reshape"([[VAR_7_]], [[VAR_3_]]) <{allowzero = 0 : si64}> : (tensor<3x8x4x64xf16>, tensor<3xi64>) -> tensor<24x4x64xf16>
// PIPE:           [[VAR_9_:%.+]] = "onnx.LayoutTransform"([[VAR_8_]]) <{target_layout = #zhigh.layout<{dataLayout = "3DS"}>}> : (tensor<24x4x64xf16>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// PIPE:           return [[VAR_9_]] : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// PIPE:         }

}

// -----

func.func @concat_expand_stick_after_canonicalize(
    %arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>)
    -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>                : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
  %scal  = onnx.Constant dense<2.000000e+00>     : tensor<f32>
  %shre  = onnx.Constant dense<[24, 8, 64]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 2 : si64}>
            : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<2x4x1x8x64xf32>, tensor<5xi64>) -> tensor<2x4x3x8x64xf32>
  %mul  = "onnx.Mul"(%exp, %scal)
            : (tensor<2x4x3x8x64xf32>, tensor<f32>) -> tensor<2x4x3x8x64xf32>
  %resh = "onnx.Reshape"(%mul, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x64xf32>, tensor<3xi64>) -> tensor<24x8x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x8x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-LABEL:  func.func @concat_expand_stick_after_canonicalize
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x3x64xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]], [[PARAM_1_]]) <{kind = "zhigh.concat-expand-stick"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<2x4x3x64xf32>, [[REGION_ARG_1_:%.+]]: tensor<2x4x5x64xf32>):
// CHECK-DAG:         [[VAR_1_:%.+]] = onnx.Constant dense<[24, 8, 64]> : tensor<3xi64>
// CHECK-DAG:         [[VAR_2_:%.+]] = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:         [[VAR_4_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:         [[VAR_5_:%.+]] = "onnx.Concat"([[REGION_ARG_0_]], [[REGION_ARG_1_]]) <{axis = 2 : si64}> : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
// CHECK:             [[VAR_6_:%.+]] = "onnx.Unsqueeze"([[VAR_5_]], [[VAR_4_]]) : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
// CHECK:             [[VAR_7_:%.+]] = "onnx.Mul"([[VAR_6_]], [[VAR_3_]]) : (tensor<2x4x1x8x64xf32>, tensor<f32>) -> tensor<2x4x1x8x64xf32>
// CHECK:             [[VAR_8_:%.+]] = "onnx.Expand"([[VAR_7_]], [[VAR_2_]]) : (tensor<2x4x1x8x64xf32>, tensor<5xi64>) -> tensor<2x4x3x8x64xf32>
// CHECK:             [[VAR_9_:%.+]] = "onnx.Reshape"([[VAR_8_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<2x4x3x8x64xf32>, tensor<3xi64>) -> tensor<24x8x64xf32>
// CHECK:             [[VAR_10_:%.+]] = "zhigh.Stick"([[VAR_9_]]) <{layout = "3DS"}> : (tensor<24x8x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:             onnx.Yield [[VAR_10_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }) {concatAxis = 2 : i64, expansionN = 3 : i64, mulScalar = 2.000000e+00 : f32, noSaturation = false, onnx_node_name = "onnx.Concat-onnx.Unsqueeze-onnx.Mul-onnx.Expand-onnx.Reshape-zhigh.Stick", reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS", unsqueezedPosition = 2 : i64, yieldConcatResult = false} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }

// PIPE-LABEL:  func.func @concat_expand_stick_after_canonicalize
// PIPE-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x3x64xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// PIPE:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]], [[PARAM_1_]]) <{kind = "zhigh.concat-expand-stick"}> ({
// PIPE:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<2x4x3x64xf32>, [[REGION_ARG_1_:%.+]]: tensor<2x4x5x64xf32>):
// PIPE-DAG:         [[VAR_1_:%.+]] = onnx.Constant dense<[24, 8, 64]> : tensor<3xi64>
// PIPE-DAG:         [[VAR_2_:%.+]] = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
// PIPE-DAG:         [[VAR_3_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// PIPE-DAG:         [[VAR_4_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// PIPE-DAG:         [[VAR_5_:%.+]] = "onnx.Concat"([[REGION_ARG_0_]], [[REGION_ARG_1_]]) <{axis = 2 : si64}> : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
// PIPE:             [[VAR_6_:%.+]] = "onnx.Unsqueeze"([[VAR_5_]], [[VAR_4_]]) : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
// PIPE:             [[VAR_7_:%.+]] = "onnx.Mul"([[VAR_6_]], [[VAR_3_]]) : (tensor<2x4x1x8x64xf32>, tensor<f32>) -> tensor<2x4x1x8x64xf32>
// PIPE:             [[VAR_8_:%.+]] = "zhigh.F32ToDLF16"([[VAR_7_]]) : (tensor<2x4x1x8x64xf32>) -> tensor<2x4x1x8x64xf16>
// PIPE:             [[VAR_9_:%.+]] = "onnx.Expand"([[VAR_8_]], [[VAR_2_]]) : (tensor<2x4x1x8x64xf16>, tensor<5xi64>) -> tensor<2x4x3x8x64xf16>
// PIPE:             [[VAR_10_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<2x4x3x8x64xf16>, tensor<3xi64>) -> tensor<24x8x64xf16>
// PIPE:             [[VAR_11_:%.+]] = "onnx.LayoutTransform"([[VAR_10_]]) <{target_layout = #zhigh.layout<{dataLayout = "3DS"}>}> : (tensor<24x8x64xf16>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// PIPE:             onnx.Yield [[VAR_11_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// PIPE:           }) {concatAxis = 2 : i64, expansionN = 3 : i64, finalLayout = "3DS", mulScalar = 2.000000e+00 : f32, noSaturation = false, onnx_node_name = "onnx.Concat-onnx.Unsqueeze-onnx.Mul-zhigh.F32ToDLF16-onnx.Expand-onnx.Reshape-onnx.LayoutTransform", reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, unsqueezedPosition = 2 : i64, yieldConcatResult = false} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// PIPE:           return [[VAR_0_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// PIPE:         }

}

