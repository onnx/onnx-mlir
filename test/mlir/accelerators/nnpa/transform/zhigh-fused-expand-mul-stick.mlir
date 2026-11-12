// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// Tests for the zhigh.expand-mul-stick FusedOp pattern.
// The pass wraps the chain:
//   ONNXUnsqueezeOp -> [ONNXMulOp] -> ONNXExpandOp -> ONNXReshapeOp -> ZHighStickOp
// into an onnx.Fused region with kind = "zhigh.expand-mul-stick".  The Mul
// step is optional; when absent, mulScalar is stored as its neutral 1.0
// default (see @expand_stick_no_mul below).  The scalar Mul is matched before
// the Expand because the Expand-sinking canonicalization moves it there before
// this pass runs; zhigh-fused-expand-mul-stick-canonicalize.mlir covers that
// interaction starting from the Expand -> Mul order.
//
// Pattern under test:
//   Input:  tensor<3x4x64xf32>
//   Unsqueeze axis 1 -> tensor<3x1x4x64xf32>   (P=1; innermost 64 is mod-64)
//   Mul by scalar 2.0  -> tensor<3x1x4x64xf32>
//   Expand dim P to N=8 -> tensor<3x8x4x64xf32>
//   Reshape [24, 4, 64] -> tensor<24x4x64xf32>  (dims 0..1 collapsed: 3*8=24)
//   Stick 3DS           -> tensor<24x4x64xf16, 3DS>
//
// Expected stored params:
//   unsqueezedPosition = 1, expansionN = 8, mulScalar = 2.0,
//   reshapeFirstCollapsedDim = 0, reshapeCollapsedCount = 2, stickFormat = "3DS"

// -----

func.func @expand_mul_stick_basic(%arg0: tensor<3x4x64xf32>)
    -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<1>             : tensor<1xi64>
  %shexp = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
  %scal  = onnx.Constant dense<2.000000e+00>  : tensor<f32>
  %shre  = onnx.Constant dense<[24, 4, 64]>   : tensor<3xi64>
  %unsq = "onnx.Unsqueeze"(%arg0, %axes)
            : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
  %mul  = "onnx.Mul"(%unsq, %scal)
            : (tensor<3x1x4x64xf32>, tensor<f32>) -> tensor<3x1x4x64xf32>
  %exp  = "onnx.Expand"(%mul, %shexp)
            : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// The fused op has exactly one external input (the function argument); all
// four constants are cloned inside the body, followed by the five chain ops.

// CHECK-LABEL:  func.func @expand_mul_stick_basic
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

}

// -----

// detectUpperCollapse — tail dim changed by reshape.
// The reshape merges dim 2 (value 4) and dim 3 (value 64) into a single dim
// of size 256, violating the invariant that dims strictly after P must be
// preserved unchanged.  detectUpperCollapse catches this at the tail check:
//   inIdx=2 (value 4) vs outIdx=1 (value 256) -> 4 != 256 -> fail.

func.func @no_fuse_tail_dim_changed(%arg0: tensor<3x4x64xf32>)
    -> tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>> {
  %axes  = onnx.Constant dense<1>             : tensor<1xi64>
  %shexp = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
  %scal  = onnx.Constant dense<2.000000e+00>  : tensor<f32>
  %shre  = onnx.Constant dense<[24, 256]>     : tensor<2xi64>
  %unsq = "onnx.Unsqueeze"(%arg0, %axes)
            : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
  %mul  = "onnx.Mul"(%unsq, %scal)
            : (tensor<3x1x4x64xf32>, tensor<f32>) -> tensor<3x1x4x64xf32>
  %exp  = "onnx.Expand"(%mul, %shexp)
            : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<3x8x4x64xf32>, tensor<2xi64>) -> tensor<24x256xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "2DS"}
            : (tensor<24x256xf32>) -> tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>>
  return %out : tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>>

// detectUpperCollapse rejects the reshape: dims after P changed.

// CHECK-LABEL:  func.func @no_fuse_tail_dim_changed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x64xf32>) -> tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[24, 256]> : tensor<2xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_2_]]) : (tensor<3x1x4x64xf32>, tensor<f32>) -> tensor<3x1x4x64xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Expand"([[VAR_5_]], [[VAR_1_]]) : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Reshape"([[VAR_6_]], [[VAR_3_]]) <{allowzero = 0 : si64}> : (tensor<3x8x4x64xf32>, tensor<2xi64>) -> tensor<24x256xf32>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Stick"([[VAR_7_]]) <{layout = "2DS"}> : (tensor<24x256xf32>) -> tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>>
// CHECK:           return [[VAR_8_]] : tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>>
// CHECK:         }

}

// -----

// detectExpandedDim — expand changes a dim other than P.
// Dim 0 changes from 3 to 6 in addition to dim P=1 changing from 1 to 8.
// detectExpandedDim catches this: non-P dim 0: 3 != 6 -> fail.

func.func @no_fuse_expand_changes_non_p_dim(%arg0: tensor<3x4x64xf32>)
    -> tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<1>             : tensor<1xi64>
  %shexp = onnx.Constant dense<[6, 8, 4, 64]> : tensor<4xi64>
  %scal  = onnx.Constant dense<2.000000e+00>  : tensor<f32>
  %shre  = onnx.Constant dense<[48, 4, 64]>   : tensor<3xi64>
  %unsq = "onnx.Unsqueeze"(%arg0, %axes)
            : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
  %mul  = "onnx.Mul"(%unsq, %scal)
            : (tensor<3x1x4x64xf32>, tensor<f32>) -> tensor<3x1x4x64xf32>
  %exp  = "onnx.Expand"(%mul, %shexp)
            : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<6x8x4x64xf32>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<6x8x4x64xf32>, tensor<3xi64>) -> tensor<48x4x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<48x4x64xf32>) -> tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// detectExpandedDim rejects the expand: non-P dim 0 changes (3 -> 6).

// CHECK-LABEL:  func.func @no_fuse_expand_changes_non_p_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x64xf32>) -> tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[6, 8, 4, 64]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[48, 4, 64]> : tensor<3xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_2_]]) : (tensor<3x1x4x64xf32>, tensor<f32>) -> tensor<3x1x4x64xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Expand"([[VAR_5_]], [[VAR_1_]]) : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<6x8x4x64xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Reshape"([[VAR_6_]], [[VAR_3_]]) <{allowzero = 0 : si64}> : (tensor<6x8x4x64xf32>, tensor<3xi64>) -> tensor<48x4x64xf32>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Stick"([[VAR_7_]]) <{layout = "3DS"}> : (tensor<48x4x64xf32>) -> tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_8_]] : tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }

}

// -----

// Mul is optional: the chain also fuses when Unsqueeze feeds Expand directly
// (no scalar multiply in between).  mulScalar stays at its neutral 1.0
// default and the fused body has only 4 ops (no onnx.Mul).

func.func @expand_stick_no_mul(%arg0: tensor<3x4x64xf32>)
    -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<1>             : tensor<1xi64>
  %shexp = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
  %shre  = onnx.Constant dense<[24, 4, 64]>   : tensor<3xi64>
  %unsq = "onnx.Unsqueeze"(%arg0, %axes)
            : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @expand_stick_no_mul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x64xf32>)
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]]) <{kind = "zhigh.expand-mul-stick"}>
// CHECK-DAG:           onnx.Constant dense<1>
// CHECK-DAG:           onnx.Constant{{.*}}[3, 8, 4, 64]
// CHECK-DAG:           onnx.Constant{{.*}}[24, 4, 64]
// Verify the four chain ops are inside the fused body, with no onnx.Mul:
// CHECK:           "onnx.Unsqueeze"{{.*}}-> tensor<3x1x4x64xf32>
// CHECK:           "onnx.Expand"{{.*}}-> tensor<3x8x4x64xf32>
// CHECK-NOT:       "onnx.Mul"
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<24x4x64xf32>
// CHECK:           "zhigh.Stick"{{.*}}-> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           onnx.Yield
// mulScalar stays at its neutral default:
// CHECK:           expansionN = 8{{.*}}mulScalar = 1.000000e+00{{.*}}reshapeCollapsedCount = 2{{.*}}reshapeFirstCollapsedDim = 0{{.*}}stickFormat = "3DS"{{.*}}unsqueezedPosition = 1
// CHECK:           return [[VAR_0_]] : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }
}

// -----

// detectIfBeneficial step 5 — stick layout not in {3D, 3DS, 4D}.
// The stick uses layout "2D", which is rejected by the layout gate.
// All earlier steps pass; the pattern fails only at the stick check.

func.func @no_fuse_stick_unsupported_layout(%arg0: tensor<64x64xf32>)
    -> tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>> {
  %axes  = onnx.Constant dense<0>              : tensor<1xi64>
  %shexp = onnx.Constant dense<[8, 64, 64]>    : tensor<3xi64>
  %scal  = onnx.Constant dense<2.000000e+00>   : tensor<f32>
  %shre  = onnx.Constant dense<[512, 64]>      : tensor<2xi64>
  %unsq = "onnx.Unsqueeze"(%arg0, %axes)
            : (tensor<64x64xf32>, tensor<1xi64>) -> tensor<1x64x64xf32>
  %mul  = "onnx.Mul"(%unsq, %scal)
            : (tensor<1x64x64xf32>, tensor<f32>) -> tensor<1x64x64xf32>
  %exp  = "onnx.Expand"(%mul, %shexp)
            : (tensor<1x64x64xf32>, tensor<3xi64>) -> tensor<8x64x64xf32>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<8x64x64xf32>, tensor<2xi64>) -> tensor<512x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "2D"}
            : (tensor<512x64xf32>) -> tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
  return %out : tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>>

// Step 5 of detectIfBeneficial rejects layout "2D": not in {3D, 3DS, 4D}.
// CHECK-LABEL:  func.func @no_fuse_stick_unsupported_layout
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<64x64xf32>) -> tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[8, 64, 64]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[512, 64]> : tensor<2xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<64x64xf32>, tensor<1xi64>) -> tensor<1x64x64xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_2_]]) : (tensor<1x64x64xf32>, tensor<f32>) -> tensor<1x64x64xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Expand"([[VAR_5_]], [[VAR_1_]]) : (tensor<1x64x64xf32>, tensor<3xi64>) -> tensor<8x64x64xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Reshape"([[VAR_6_]], [[VAR_3_]]) <{allowzero = 0 : si64}> : (tensor<8x64x64xf32>, tensor<2xi64>) -> tensor<512x64xf32>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Stick"([[VAR_7_]]) <{layout = "2D"}> : (tensor<512x64xf32>) -> tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:           return [[VAR_8_]] : tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
// CHECK:         }

}

// -----

// The Mul's other operand is not a constant, so it is not absorbed as the
// optional scalar Mul; the Expand is then not the Unsqueeze's single user, and
// nothing fuses.

func.func @no_fuse_mul_by_non_constant(%arg0: tensor<3x4x64xf32>, %arg1: tensor<f32>)
    -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<1>             : tensor<1xi64>
  %shexp = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
  %shre  = onnx.Constant dense<[24, 4, 64]>   : tensor<3xi64>
  %unsq = "onnx.Unsqueeze"(%arg0, %axes)
            : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
  %mul  = "onnx.Mul"(%unsq, %arg1)
            : (tensor<3x1x4x64xf32>, tensor<f32>) -> tensor<3x1x4x64xf32>
  %exp  = "onnx.Expand"(%mul, %shexp)
            : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @no_fuse_mul_by_non_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x64xf32>, [[PARAM_1_:%.+]]: tensor<f32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[24, 4, 64]> : tensor<3xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_3_]], [[PARAM_1_]]) : (tensor<3x1x4x64xf32>, tensor<f32>) -> tensor<3x1x4x64xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Expand"([[VAR_4_]], [[VAR_1_]]) : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Reshape"([[VAR_5_]], [[VAR_2_]]) <{allowzero = 0 : si64}> : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
// CHECK:           [[VAR_7_:%.+]] = "zhigh.Stick"([[VAR_6_]]) <{layout = "3DS"}> : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_7_]] : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }

}

// -----

// Only the Mul-before-Expand order is matched.  A scalar Mul after the Expand
// is no longer absorbed by this pass alone (in the compiler pipeline, the
// Expand-sinking canonicalization has already moved it before the Expand), so
// the chain stays unfused here.

func.func @no_fuse_mul_after_expand(%arg0: tensor<3x4x64xf32>)
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
// CHECK-LABEL:  func.func @no_fuse_mul_after_expand
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[24, 4, 64]> : tensor<3xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Expand"([[VAR_4_]], [[VAR_1_]]) : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Mul"([[VAR_5_]], [[VAR_2_]]) : (tensor<3x8x4x64xf32>, tensor<f32>) -> tensor<3x8x4x64xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Reshape"([[VAR_6_]], [[VAR_3_]]) <{allowzero = 0 : si64}> : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
// CHECK:           [[VAR_8_:%.+]] = "zhigh.Stick"([[VAR_7_]]) <{layout = "3DS"}> : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_8_]] : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:         }

}

