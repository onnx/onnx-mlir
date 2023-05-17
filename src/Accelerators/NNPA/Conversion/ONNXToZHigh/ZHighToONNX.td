// SPDX-License-Identifier: Apache-2.0

//===- ZHighToONNX.td - Replacing ZHigh Ops by ONNX Ops -*- tablegen ------===//
//
// Copyright 2023- The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ZHIGH_TO_ONNX
#define ZHIGH_TO_ONNX

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNX.td"
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// ONNXAddOp %X = ZHighUnstickOp (ZHighAddOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighAddPattern : Pat<
  (ZHighUnstickOp (ZHighAddOp (ZHighStickOp $x, $l1), (ZHighStickOp $y, $l2))),
  (ONNXAddOp $x, $y)
>;


//===----------------------------------------------------------------------===//
// ONNXMulOp %X = ZHighUnstickOp (ZHighMulOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighMulPattern : Pat<
  (ZHighUnstickOp (ZHighMulOp (ZHighStickOp $x, $_), (ZHighStickOp $y, $_))),
  (ONNXMulOp $x, $y)
>;

//===----------------------------------------------------------------------===//
// ONNXSubOp %X = ZHighUnstickOp (ZHighSubOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighSubPattern : Pat<
  (ZHighUnstickOp (ZHighSubOp (ZHighStickOp $x, $_), (ZHighStickOp $y, $_))),
  (ONNXSubOp $x, $y)
>;

//===----------------------------------------------------------------------===//
// ONNXDivOp %X = ZHighUnstickOp (ZHighDivOp (ZHighStickOp
// %X),(ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighDivPattern : Pat<
  (ZHighUnstickOp (ZHighDivOp (ZHighStickOp $x, $_), (ZHighStickOp $y, $_))),
  (ONNXDivOp $x, $y)
>;

//===----------------------------------------------------------------------===//
// ONNXLogOp %X = ZHighUnstickOp (ZHighLogOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceZHighLogPattern : Pat<
  (ZHighUnstickOp (ZHighLogOp (ZHighStickOp $x, $_))),
  (ONNXLogOp $x)
>;

//===----------------------------------------------------------------------===//
// ONNXReluOp %X = ZHighUnstickOp (ZHighReluOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceZHighReluPattern : Pat<
  (ZHighUnstickOp (ZHighReluOp (ZHighStickOp $x, $_))),
  (ONNXReluOp $x)
>;

#endif // ZHIGH_TO_ONNX
