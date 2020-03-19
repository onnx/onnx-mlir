//===- ONNXDecompose.td - Rewriting for decomposing ONNX Ops -*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_DECOMPOSE
#define ONNX_DECOMPOSE

#ifndef OP_BASE
include "Dialect/ONNX/ONNXOps.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// ONNXReduceL1Op %X = ONNXReduceSumOp (ONNXAbsOp %X)
//===----------------------------------------------------------------------===//
def ReduceL1OpPattern: Pat<(ONNXReduceL1Op $oprd, $axes, $keepdims),
                           (ONNXReduceSumOp (ONNXAbsOp $oprd), $axes, $keepdims)>;

//===----------------------------------------------------------------------===//
// ONNXReduceL2Op %X = ONNXSqrtOp (ONNXReduceSumSquareOp (%X))
//===----------------------------------------------------------------------===//
def ReduceL2OpPattern: Pat<(ONNXReduceL2Op $oprd, $axes, $keepdims),
                           (ONNXSqrtOp (ONNXReduceSumSquareOp $oprd, $axes, $keepdims))>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumOp %X = ONNXLogOp (ONNXReduceSumOp (%X))
//===----------------------------------------------------------------------===//
def ReduceLogSumOpPattern: Pat<(ONNXReduceLogSumOp $oprd, $axes, $keepdims),
                           (ONNXLogOp (ONNXReduceSumOp $oprd, $axes, $keepdims))>;

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumExpOp %X = ONNXReduceLogSumOp (ONNXExpOp %X)
//===----------------------------------------------------------------------===//
def ReduceLogSumExpOpPattern: Pat<(ONNXReduceLogSumExpOp $oprd, $axes, $keepdims),
                                  (ONNXReduceLogSumOp (ONNXExpOp $oprd), $axes, $keepdims)>;

//===----------------------------------------------------------------------===//
// ONNXReduceSumSquareOp %X = ONNXReduceSumOp (ONNXMulOp %X, %X)
//===----------------------------------------------------------------------===//
def ReduceSumSquareOpPattern: Pat<(ONNXReduceSumSquareOp $oprd, $axes, $keepdims),
                                  (ONNXReduceSumOp (ONNXMulOp $oprd, $oprd), $axes, $keepdims)>;

#endif // ONNX_DECOMPOSE
