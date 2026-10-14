// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// Tests for the zhigh.expand-mul-stick FusedOp pattern.
// The pass wraps the chain:
//   ONNXUnsqueezeOp -> ONNXExpandOp -> [ONNXMulOp] -> ONNXReshapeOp -> ZHighStickOp
// into an onnx.Fused region with kind = "zhigh.expand-mul-stick".  The Mul
// step is optional; when absent, mulScalar is stored as its neutral 1.0
// default (see @expand_stick_no_mul below).
//
// Pattern under test:
//   Input:  tensor<3x4x64xf32>
//   Unsqueeze axis 1 -> tensor<3x1x4x64xf32>   (P=1; innermost 64 is mod-64)
//   Expand dim P to N=8 -> tensor<3x8x4x64xf32>
//   Mul by scalar 2.0  -> tensor<3x8x4x64xf32>
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
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
  %mul  = "onnx.Mul"(%exp, %scal)
            : (tensor<3x8x4x64xf32>, tensor<f32>) -> tensor<3x8x4x64xf32>
  %resh = "onnx.Reshape"(%mul, %shre) <{allowzero = 0 : si64}>
            : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @expand_mul_stick_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x64xf32>)
// Fused op is created with exactly one external input (the function argument);
// all four constants are cloned inside the body.
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]]) <{kind = "zhigh.expand-mul-stick"}>
// Constants are cloned inside the fused body:
// CHECK-DAG:           onnx.Constant dense<1>
// CHECK-DAG:           onnx.Constant{{.*}}[3, 8, 4, 64]
// CHECK-DAG:           onnx.Constant{{.*}}2.000000e+00
// CHECK-DAG:           onnx.Constant{{.*}}[24, 4, 64]
// Verify the five chain ops are inside the fused body:
// CHECK:           "onnx.Unsqueeze"{{.*}}-> tensor<3x1x4x64xf32>
// CHECK:           "onnx.Expand"{{.*}}-> tensor<3x8x4x64xf32>
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<24x4x64xf32>
// CHECK:           "zhigh.Stick"{{.*}}-> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           onnx.Yield
// Verify stored params (attrs printed alphabetically after the body closes):
// CHECK:           expansionN = 8{{.*}}mulScalar = 2.000000e+00{{.*}}reshapeCollapsedCount = 2{{.*}}reshapeFirstCollapsedDim = 0{{.*}}stickFormat = "3DS"{{.*}}unsqueezedPosition = 1
// CHECK:           return [[VAR_0_]] : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }
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
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
  %mul  = "onnx.Mul"(%exp, %scal)
            : (tensor<3x8x4x64xf32>, tensor<f32>) -> tensor<3x8x4x64xf32>
  %resh = "onnx.Reshape"(%mul, %shre) <{allowzero = 0 : si64}>
            : (tensor<3x8x4x64xf32>, tensor<2xi64>) -> tensor<24x256xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "2DS"}
            : (tensor<24x256xf32>) -> tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>>
  return %out : tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>>

// CHECK-LABEL:  func.func @no_fuse_tail_dim_changed
// detectUpperCollapse rejects the reshape: dims after P changed.
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Unsqueeze"
// CHECK:           "onnx.Expand"
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.Reshape"
// CHECK:           "zhigh.Stick"
// CHECK:           return
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
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<6x8x4x64xf32>
  %mul  = "onnx.Mul"(%exp, %scal)
            : (tensor<6x8x4x64xf32>, tensor<f32>) -> tensor<6x8x4x64xf32>
  %resh = "onnx.Reshape"(%mul, %shre) <{allowzero = 0 : si64}>
            : (tensor<6x8x4x64xf32>, tensor<3xi64>) -> tensor<48x4x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "3DS"}
            : (tensor<48x4x64xf32>) -> tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @no_fuse_expand_changes_non_p_dim
// detectExpandedDim rejects the expand: non-P dim 0 changes (3 -> 6).
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Unsqueeze"
// CHECK:           "onnx.Expand"
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.Reshape"
// CHECK:           "zhigh.Stick"
// CHECK:           return
}

// -----

// Mul is optional: the chain also fuses when Expand feeds Reshape directly
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
  %exp  = "onnx.Expand"(%unsq, %shexp)
            : (tensor<1x64x64xf32>, tensor<3xi64>) -> tensor<8x64x64xf32>
  %mul  = "onnx.Mul"(%exp, %scal)
            : (tensor<8x64x64xf32>, tensor<f32>) -> tensor<8x64x64xf32>
  %resh = "onnx.Reshape"(%mul, %shre) <{allowzero = 0 : si64}>
            : (tensor<8x64x64xf32>, tensor<2xi64>) -> tensor<512x64xf32>
  %out  = "zhigh.Stick"(%resh) {layout = "2D"}
            : (tensor<512x64xf32>) -> tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
  return %out : tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>>

// CHECK-LABEL:  func.func @no_fuse_stick_unsupported_layout
// Step 5 of detectIfBeneficial rejects layout "2D": not in {3D, 3DS, 4D}.
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Unsqueeze"
// CHECK:           "onnx.Expand"
// CHECK:           "onnx.Mul"
// CHECK:           "onnx.Reshape"
// CHECK:           "zhigh.Stick"
// CHECK:           return
}
