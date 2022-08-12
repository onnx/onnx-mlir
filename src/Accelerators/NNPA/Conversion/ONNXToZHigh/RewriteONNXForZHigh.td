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
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td"
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

//===----------------------------------------------------------------------===//
// Rewrite AddOp if one of its inputs is a constant and the constant is
// unidirectional broadcastable to X.
// For example, X's shape is [1x4x8] and constant's shape is [8].
// 
// This is to eliminate broadcasting that is not supported by NNPA.
//
// ONNXAddOp %X, %constant = ONNXAddOp %X, (ONNXExpandOp %constant, (ONNXShapeOp %X))
//
// Assuming that ONNXAddOp has been canonicalized to the form of (X + constant)
// 
//===----------------------------------------------------------------------===//

// Get a type for a tensor that stores the shape of another tensor.
def GetShapeTypeOf: NativeCodeCall<
  "RankedTensorType::get({$0.getType().cast<ShapedType>().getRank()}, $_builder.getIntegerType(64))"
>;

// Check unidirectional broadcasting from the first to second tensor.
def IsUniBroadcastingFromFirstToSecond: Constraint<
  CPred<"isUniBroadcatableFirstToSecond($0, $1)">,
  "Is unidirectional broadcasting from the first to second tensor"
>;

def IsFromONNXConstantOp: Constraint<
    CPred<"llvm::dyn_cast_or_null<ONNXConstantOp>($0.getDefiningOp())">,
    "Is a value from ONNXConstantOp">;
def expandConstantOperandForAddOp: Pat<
  (ONNXAddOp $x, (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_)),
  (ONNXAddOp $x, (ONNXExpandOp $c,
                               (ONNXShapeOp $x, (returnType (GetShapeTypeOf $x))),
                               (returnType $x))),
  [(IsUniBroadcastingFromFirstToSecond $c, $x)]
>;

#endif // REWRITE_ONNX_FOR_ZHIGH
