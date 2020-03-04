//===----------------------------------------------------------------------===//
//=- onnx_rewrite.td - Pattern Match Rewriting for ONNX -*- tablegen -*----===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//

#ifndef ONNX_REWRITE
#define ONNX_REWRITE

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

#endif // ONNX_REWRITE
