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

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// onnx.add(onnx.matmult(%X, %Y), %Z) = onnx.gemm(%X, %Y, %Z)
def MulAddToGemmOptPattern : Pat<(ONNXAddOp(ONNXMatMulOp $m1, $m2), $m3),
                                 (ONNXFullGemmOp $m1, $m2, $m3)>;

#endif // ONNX_COMBINE
