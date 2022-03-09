// SPDX-License-Identifier: Apache-2.0

//===- RewriteONNXForZHigh.td - Rewrite ONNX Ops for ZHigh -*- tablegen ---===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef REWRITE_ONNX_FOR_ZHIGH
#define REWRITE_ONNX_FOR_ZHIGH

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.td"
include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

def GetSqrtResultBatchNormA :
      NativeCodeCall<"getSqrtResultBatchNormA($_loc, $_builder, $0, $1)">;

//===----------------------------------------------------------------------===//
// Rewrite
//
// ONNXBatchNormalizationInferenceModeOp %X
//             = NHWCtoNCHW
//                 (ZHighUnstickOp
//                     (ZHighBatchNormOp
//                         (ZHighStickOp
//                             (NCHWtoNHWC %X)),
//                         (ZHighStickOp %A),
//                         (ZHighStickOp %B)))
//
// %A = $scale / sqrt($var + $epsilon))
// %B = $b - $mean * %A
//===----------------------------------------------------------------------===//
def replaceONNXBatchNormalizationInferenceModePattern : Pattern<
  (ONNXBatchNormalizationInferenceModeOp:$res $x, $scale, $b, $mean, $var,
                                         $epsilon, $momentum),
  [
    // $A = scale / sqrt(var + epsilon)
    (ONNXDivOp:$A $scale, (GetSqrtResultBatchNormA $var, $epsilon)),
    // $B = bias - mean * $A
    (ONNXSubOp:$B $b, (ONNXMulOp $mean, $A)),

    // Calculate BatchNorm Op using $A and $B
    (NHWCtoNCHW (GetTypeOf $res),
        (ZHighUnstickOp
            (ZHighBatchNormOp
                (ZHighStickOp (NCHWtoNHWC $x), (NHWCLayoutAttr)),
                (ZHighStickOp $A, (_1DLayoutAttr)),
                (ZHighStickOp $B, (_1DLayoutAttr)))))
  ]
>;

#endif // REWRITE_ONNX_FOR_ZHIGH
