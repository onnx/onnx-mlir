//=- ONNXCombine.td - Pattern Match Optimizations for ONNX -*- tablegen -*-===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_COMBINE
#define ONNX_COMBINE

#ifndef OP_BASE
include "dialect/onnx/onnx.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

def HasOneUse : Constraint<CPred<"$0->hasOneUse()">>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// onnx.add(onnx.matmul(%X, %Y), %Z) = onnx.FullGemm(%X, %Y, %Z)
def MulAddToGemmOptPattern : Pat<(ONNXAddOp (ONNXMatMulOp:$res $m1, $m2), $m3),
                                 (ONNXFullGemmOp $m1, $m2, $m3),
				 [(HasOneUse $res)]>;

// ONNX_Op (onnx.Identity (%X)) = ONNX_Op (%X)
def IdentityEliminationPattern : Pat<(ONNXIdentityOp $arg),
				     (replaceWithValue $arg)>;

#endif // ONNX_COMBINE
