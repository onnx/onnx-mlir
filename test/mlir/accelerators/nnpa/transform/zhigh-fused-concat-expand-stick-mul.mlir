// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// Tests for the "1-step" explicit-stick tail of the zhigh.concat-expand-stick
// FusedOp pattern (the Granite KV-repeat-and-scale idiom): unlike
// zhigh-fused-concat-expand-stick.mlir's 2-step tail (F32ToDLF16 +
// LayoutTransform), this tail ends directly in a ZHighStickOp and allows an
// optional scalar Mul (the attention scale factor) between Expand and
// Reshape. The pass wraps the chain:
//   ONNXConcatOp -> ONNXUnsqueezeOp -> ONNXExpandOp -> [ONNXMulOp] ->
//   ONNXReshapeOp -> ZHighStickOp
// into a single onnx.Fused region with kind = "zhigh.concat-expand-stick",
// same as the 2-step tail -- exactly one of finalLayout / stickFormat is set
// depending on which tail matched; here it's always stickFormat.
//
// Pattern under test (@concat_expand_stick_mul_basic):
//   Inputs: tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>
//   Concat axis 2          -> tensor<2x4x8x64xf32>
//   Unsqueeze axis 2       -> tensor<2x4x1x8x64xf32>  (P=2)
//   Expand dim P to N=3    -> tensor<2x4x3x8x64xf32>
//   Mul by scalar 2.0      -> tensor<2x4x3x8x64xf32>
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
// Unsqueeze->Expand->Mul->Reshape->Stick tail must still fuse on its own as
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
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<2x4x1x8x64xf32>, tensor<5xi64>) -> tensor<2x4x3x8x64xf32>
  %mul  = "onnx.Mul"(%exp, %scal)
            : (tensor<2x4x3x8x64xf32>, tensor<f32>) -> tensor<2x4x3x8x64xf32>
  %resh = "onnx.Reshape"(%mul, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x64xf32>, tensor<3xi64>) -> tensor<24x8x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x8x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @concat_expand_stick_mul_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x3x64xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5x64xf32>)
// Fused op is created with exactly two external inputs (the function
// arguments); the constants are cloned inside the body.
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]], [[PARAM_1_]]) <{kind = "zhigh.concat-expand-stick"}>
// Verify the six chain ops are inside the fused body, ending in an explicit
// ZHighStickOp rather than F32ToDLF16 + LayoutTransform:
// CHECK:           "onnx.Concat"{{.*}}-> tensor<2x4x8x64xf32>
// CHECK:           "onnx.Unsqueeze"{{.*}}-> tensor<2x4x1x8x64xf32>
// CHECK:           "onnx.Expand"{{.*}}-> tensor<2x4x3x8x64xf32>
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<24x8x64xf32>
// CHECK:           "zhigh.Stick"{{.*}}-> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           onnx.Yield
// Verify stored params (attrs printed alphabetically after the body closes).
// stickFormat is set (not finalLayout), and mulScalar reflects the Mul step:
// CHECK:           concatAxis = 2{{.*}}expansionN = 3{{.*}}mulScalar = 2.000000e+00{{.*}}reshapeCollapsedCount = 3{{.*}}reshapeFirstCollapsedDim = 0{{.*}}stickFormat = "3DS"{{.*}}unsqueezedPosition = 2{{.*}}yieldConcatResult = false
// CHECK:           return [[VAR_0_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }
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
// Unsqueeze->Expand->Mul->Reshape->Stick tail must still fuse on its own via
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
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<2x4x1x3x64xf32>, tensor<5xi64>) -> tensor<2x4x3x3x64xf32>
  %mul  = "onnx.Mul"(%exp, %scal)
            : (tensor<2x4x3x3x64xf32>, tensor<f32>) -> tensor<2x4x3x3x64xf32>
  %resh = "onnx.Reshape"(%mul, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x3x64xf32>, tensor<3xi64>) -> tensor<24x3x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x3x64xf32>) -> tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @concat_expand_stick_precedence_regression
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x1x64xf32>, [[PARAM_1_:%.+]]: tensor<2x4x1x64xf32>, [[PARAM_2_:%.+]]: tensor<2x4x1x64xf32>)
// The 3-input Concat stays a plain, unfused op:
// CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) <{axis = 2 : si64}> : (tensor<2x4x1x64xf32>, tensor<2x4x1x64xf32>, tensor<2x4x1x64xf32>) -> tensor<2x4x3x64xf32>
// ...and the rest of the chain still fuses on its own, taking the Concat's
// result as its single external input:
// CHECK:           [[VAR_1_:%.+]] = "onnx.Fused"([[VAR_0_]]) <{kind = "zhigh.expand-mul-stick"}>
// CHECK:           "onnx.Unsqueeze"{{.*}}-> tensor<2x4x1x3x64xf32>
// CHECK:           "onnx.Expand"{{.*}}-> tensor<2x4x3x3x64xf32>
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<24x3x64xf32>
// CHECK:           "zhigh.Stick"{{.*}}-> tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           onnx.Yield
// CHECK:           expansionN = 3{{.*}}mulScalar = 2.000000e+00{{.*}}reshapeCollapsedCount = 3{{.*}}reshapeFirstCollapsedDim = 0{{.*}}stickFormat = "3DS"{{.*}}unsqueezedPosition = 2
// CHECK:           return [[VAR_1_]] : tensor<24x3x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }
}
