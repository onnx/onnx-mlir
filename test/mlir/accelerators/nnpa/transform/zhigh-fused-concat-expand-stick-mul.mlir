// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// Tests for the "1-step" explicit-stick tail of the zhigh.concat-expand-stick
// FusedOp pattern (the Granite KV-repeat-and-scale idiom): unlike
// zhigh-fused-concat-expand-stick.mlir's 2-step tail (F32ToDLF16 +
// LayoutTransform), this tail ends directly in a ZHighStickOp and allows an
// optional scalar Mul (the attention scale factor) between Unsqueeze and
// Expand. The pass wraps the chain:
//   ONNXConcatOp -> ONNXUnsqueezeOp -> [ONNXMulOp] -> ONNXExpandOp ->
//   ONNXReshapeOp -> ZHighStickOp
// into a single onnx.Fused region with kind = "zhigh.concat-expand-stick",
// same as the 2-step tail -- exactly one of finalLayout / stickFormat is set
// depending on which tail matched; here it's always stickFormat. The scalar
// Mul is matched before the Expand because the Expand-sinking
// canonicalization moves it there before this pass runs;
// zhigh-fused-expand-mul-stick-canonicalize.mlir covers that interaction
// starting from the Expand -> Mul order.
//
// Pattern under test (@concat_expand_stick_mul_basic):
//   Inputs: tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>
//   Concat axis 2          -> tensor<2x4x8x64xf32>
//   Unsqueeze axis 2       -> tensor<2x4x1x8x64xf32>  (P=2)
//   Mul by scalar 2.0      -> tensor<2x4x1x8x64xf32>
//   Expand dim P to N=3    -> tensor<2x4x3x8x64xf32>
//   Reshape [24, 8, 64]    -> tensor<24x8x64xf32>      (dims 0..2 collapsed)
//   Stick 3DS              -> tensor<24x8x64xf16, 3DS>
//
// Expected stored params:
//   concatAxis = 2, unsqueezedPosition = 2, expansionN = 3,
//   mulScalar = 2.0, reshapeFirstCollapsedDim = 0, reshapeCollapsedCount = 3,
//   stickFormat = "3DS", yieldConcatResult = false
//
// @concat_expand_stick_no_mul_basic is the same chain without the Mul step
// (mulScalar stays at its neutral 1.0 default -- see
// ExpandMulStickFusionHelper's own no-mul case for the same convention).
//
// @concat_expand_stick_precedence_regression is a precedence regression
// guard: when the Concat itself is not a viable head for this fusion (here,
// 3 inputs instead of 2), the Concat must stay unfused and the
// Unsqueeze->Mul->Expand->Reshape->Stick tail must still fuse on its own as
// a standalone zhigh.expand-mul-stick op -- confirming that always running
// FusedPatternsForConcatExpandStick in its own, earlier pass phase (see
// FusionOpStickUnstick.cpp) does not suppress ExpandMulStickFusionHelper's
// own fusion when Concat can't claim the chain.

// -----

func.func @concat_expand_stick_mul_basic(
    %arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>)
    -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>               : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
  %scal  = onnx.Constant dense<2.000000e+00>    : tensor<f32>
  %shre  = onnx.Constant dense<[24, 8, 64]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 2 : si64}>
            : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
  %mul  = "onnx.Mul"(%unsq, %scal)
            : (tensor<2x4x1x8x64xf32>, tensor<f32>) -> tensor<2x4x1x8x64xf32>
  %exp  = "onnx.Expand"(%mul, %shexp)
            : (tensor<2x4x1x8x64xf32>, tensor<5xi64>) -> tensor<2x4x3x8x64xf32>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x64xf32>, tensor<3xi64>) -> tensor<24x8x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x8x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// The fused op has exactly two external inputs (the function arguments); the
// constants are cloned inside the body, followed by the six chain ops, ending
// in an explicit ZHighStickOp rather than F32ToDLF16 + LayoutTransform.
// stickFormat is set (not finalLayout), and mulScalar reflects the Mul step.

// CHECK-LABEL:  func.func @concat_expand_stick_mul_basic
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

}

// -----

func.func @concat_expand_stick_no_mul_basic(
    %arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>)
    -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>               : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
  %shre  = onnx.Constant dense<[24, 8, 64]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 2 : si64}>
            : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<2x4x1x8x64xf32>, tensor<5xi64>) -> tensor<2x4x3x8x64xf32>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x64xf32>, tensor<3xi64>) -> tensor<24x8x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x8x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @concat_expand_stick_no_mul_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x3x64xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5x64xf32>)
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]], [[PARAM_1_]]) <{kind = "zhigh.concat-expand-stick"}>
// No onnx.Mul in the body this time:
// CHECK:           "onnx.Concat"{{.*}}-> tensor<2x4x8x64xf32>
// CHECK:           "onnx.Unsqueeze"{{.*}}-> tensor<2x4x1x8x64xf32>
// CHECK:           "onnx.Expand"{{.*}}-> tensor<2x4x3x8x64xf32>
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<24x8x64xf32>
// CHECK:           "zhigh.Stick"{{.*}}-> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           onnx.Yield
// mulScalar is the neutral 1.0 default:
// CHECK:           concatAxis = 2{{.*}}expansionN = 3{{.*}}mulScalar = 1.000000e+00{{.*}}reshapeCollapsedCount = 3{{.*}}reshapeFirstCollapsedDim = 0{{.*}}stickFormat = "3DS"{{.*}}unsqueezedPosition = 2{{.*}}yieldConcatResult = false
// CHECK:           return [[VAR_0_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }
}

// -----

// Precedence regression guard: the Concat has three inputs, so it fails
// ConcatExpandStickFusionHelper's own Step-1 viability check regardless of
// which phase runs first -- it must stay unfused, and the
// Unsqueeze->Mul->Expand->Reshape->Stick tail must still fuse on its own via
// ExpandMulStickFusionHelper.

func.func @concat_expand_stick_precedence_regression(
    %arg0: tensor<2x4x1x64xf32>, %arg1: tensor<2x4x1x64xf32>, %arg2: tensor<2x4x1x64xf32>)
    -> tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>                : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 4, 3, 3, 64]> : tensor<5xi64>
  %scal  = onnx.Constant dense<2.000000e+00>     : tensor<f32>
  %shre  = onnx.Constant dense<[24, 3, 64]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1, %arg2) <{axis = 2 : si64}>
            : (tensor<2x4x1x64xf32>, tensor<2x4x1x64xf32>, tensor<2x4x1x64xf32>) -> tensor<2x4x3x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x4x3x64xf32>, tensor<1xi64>) -> tensor<2x4x1x3x64xf32>
  %mul  = "onnx.Mul"(%unsq, %scal)
            : (tensor<2x4x1x3x64xf32>, tensor<f32>) -> tensor<2x4x1x3x64xf32>
  %exp  = "onnx.Expand"(%mul, %shexp)
            : (tensor<2x4x1x3x64xf32>, tensor<5xi64>) -> tensor<2x4x3x3x64xf32>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x3x64xf32>, tensor<3xi64>) -> tensor<24x3x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x3x64xf32>) -> tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// The 3-input Concat stays a plain, unfused op, and the rest of the chain
// still fuses on its own, taking the Concat's result as its single external
// input.
// CHECK-LABEL:  func.func @concat_expand_stick_precedence_regression
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x1x64xf32>, [[PARAM_1_:%.+]]: tensor<2x4x1x64xf32>, [[PARAM_2_:%.+]]: tensor<2x4x1x64xf32>) -> tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) <{axis = 2 : si64}> : (tensor<2x4x1x64xf32>, tensor<2x4x1x64xf32>, tensor<2x4x1x64xf32>) -> tensor<2x4x3x64xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Fused"([[VAR_0_]]) <{kind = "zhigh.expand-mul-stick"}> ({
// CHECK:           ^bb0([[REGION_ARG_0_:%.+]]: tensor<2x4x3x64xf32>):
// CHECK-DAG:         [[VAR_2_:%.+]] = onnx.Constant dense<[24, 3, 64]> : tensor<3xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = onnx.Constant dense<[2, 4, 3, 3, 64]> : tensor<5xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:         [[VAR_5_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK:             [[VAR_6_:%.+]] = "onnx.Unsqueeze"([[REGION_ARG_0_]], [[VAR_5_]]) : (tensor<2x4x3x64xf32>, tensor<1xi64>) -> tensor<2x4x1x3x64xf32>
// CHECK:             [[VAR_7_:%.+]] = "onnx.Mul"([[VAR_6_]], [[VAR_4_]]) : (tensor<2x4x1x3x64xf32>, tensor<f32>) -> tensor<2x4x1x3x64xf32>
// CHECK:             [[VAR_8_:%.+]] = "onnx.Expand"([[VAR_7_]], [[VAR_3_]]) : (tensor<2x4x1x3x64xf32>, tensor<5xi64>) -> tensor<2x4x3x3x64xf32>
// CHECK:             [[VAR_9_:%.+]] = "onnx.Reshape"([[VAR_8_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<2x4x3x3x64xf32>, tensor<3xi64>) -> tensor<24x3x64xf32>
// CHECK:             [[VAR_10_:%.+]] = "zhigh.Stick"([[VAR_9_]]) <{layout = "3DS"}> : (tensor<24x3x64xf32>) -> tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:             onnx.Yield [[VAR_10_]] : tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }) {expansionN = 3 : i64, mulScalar = 2.000000e+00 : f32, onnx_node_name = "onnx.Unsqueeze-onnx.Mul-onnx.Expand-onnx.Reshape-zhigh.Stick", reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS", unsqueezedPosition = 2 : i64} : (tensor<2x4x3x64xf32>) -> tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_1_]] : tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }

}


// -----

// 2-step (LayoutTransform) tail with the scalar Mul: this is the form the
// granite-4 K path reaches the fusion in, once the Expand-sinking
// canonicalization has moved the Mul above the Expand and
// ZHighDecomposeStickUnstick has split the Stick into F32ToDLF16 (placed
// before the Expand, after the Mul) and LayoutTransform. The Mul is in f32,
// before the conversion. Fuses into one op with mulScalar = 2.0 and
// finalLayout = "3DS" (no stickFormat).

func.func @concat_expand_stick_mul_layout_transform(
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
  %mul  = "onnx.Mul"(%unsq, %scal)
            : (tensor<2x4x1x8x64xf32>, tensor<f32>) -> tensor<2x4x1x8x64xf32>
  %dlf  = "zhigh.F32ToDLF16"(%mul)
            : (tensor<2x4x1x8x64xf32>) -> tensor<2x4x1x8x64xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<2x4x1x8x64xf16>, tensor<5xi64>) -> tensor<2x4x3x8x64xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x64xf16>, tensor<3xi64>) -> tensor<24x8x64xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>}
            : (tensor<24x8x64xf16>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @concat_expand_stick_mul_layout_transform
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
// CHECK:             [[VAR_8_:%.+]] = "zhigh.F32ToDLF16"([[VAR_7_]]) : (tensor<2x4x1x8x64xf32>) -> tensor<2x4x1x8x64xf16>
// CHECK:             [[VAR_9_:%.+]] = "onnx.Expand"([[VAR_8_]], [[VAR_2_]]) : (tensor<2x4x1x8x64xf16>, tensor<5xi64>) -> tensor<2x4x3x8x64xf16>
// CHECK:             [[VAR_10_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_1_]]) <{allowzero = 0 : si64}> : (tensor<2x4x3x8x64xf16>, tensor<3xi64>) -> tensor<24x8x64xf16>
// CHECK:             [[VAR_11_:%.+]] = "onnx.LayoutTransform"([[VAR_10_]]) <{target_layout = #zhigh.layout<{dataLayout = "3DS"}>}> : (tensor<24x8x64xf16>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:             onnx.Yield [[VAR_11_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }) {concatAxis = 2 : i64, expansionN = 3 : i64, finalLayout = "3DS", mulScalar = 2.000000e+00 : f32, noSaturation = false, onnx_node_name = "onnx.Concat-onnx.Unsqueeze-onnx.Mul-zhigh.F32ToDLF16-onnx.Expand-onnx.Reshape-onnx.LayoutTransform", reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, unsqueezedPosition = 2 : i64, yieldConcatResult = false} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }

}

// -----

// Negative: the same 2-step chain, but the Mul's second operand is a
// function argument, not a constant scalar. detectOptionalScalarMul rejects
// it, the Unsqueeze's user is then not an F32ToDLF16 either, and nothing
// fuses.

func.func @concat_expand_stick_mul_layout_transform_nonconst(
    %arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>, %arg2: tensor<f32>)
    -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>                : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
  %shre  = onnx.Constant dense<[24, 8, 64]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 2 : si64}>
            : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
  %mul  = "onnx.Mul"(%unsq, %arg2)
            : (tensor<2x4x1x8x64xf32>, tensor<f32>) -> tensor<2x4x1x8x64xf32>
  %dlf  = "zhigh.F32ToDLF16"(%mul)
            : (tensor<2x4x1x8x64xf32>) -> tensor<2x4x1x8x64xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<2x4x1x8x64xf16>, tensor<5xi64>) -> tensor<2x4x3x8x64xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x64xf16>, tensor<3xi64>) -> tensor<24x8x64xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>}
            : (tensor<24x8x64xf16>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-LABEL:  func.func @concat_expand_stick_mul_layout_transform_nonconst
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x3x64xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5x64xf32>, [[PARAM_2_:%.+]]: tensor<f32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[24, 8, 64]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]]) <{axis = 2 : si64}> : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[VAR_3_]], [[VAR_0_]]) : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[PARAM_2_]]) : (tensor<2x4x1x8x64xf32>, tensor<f32>) -> tensor<2x4x1x8x64xf32>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.F32ToDLF16"([[VAR_5_]]) : (tensor<2x4x1x8x64xf32>) -> tensor<2x4x1x8x64xf16>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Expand"([[VAR_6_]], [[VAR_1_]]) : (tensor<2x4x1x8x64xf16>, tensor<5xi64>) -> tensor<2x4x3x8x64xf16>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Reshape"([[VAR_7_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<2x4x3x8x64xf16>, tensor<3xi64>) -> tensor<24x8x64xf16>
// CHECK:           [[VAR_9_:%.+]] = "onnx.LayoutTransform"([[VAR_8_]]) <{target_layout = #zhigh.layout<{dataLayout = "3DS"}>}> : (tensor<24x8x64xf16>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_9_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }

}

