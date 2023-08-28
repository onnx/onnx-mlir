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
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

def NotBlockArgument : Constraint<CPred<"!isa<BlockArgument>($_self)">>;
def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">>;

//===----------------------------------------------------------------------===//
// ONNXAddOp %X = ZHighUnstickOp (ZHighAddOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighAddPattern : Pat<
  (ZHighUnstickOp (ZHighAddOp (ZHighStickOp:$s_x $x, $_), (ZHighStickOp:$s_y $y, $_))),
  (ONNXAddOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (HasOneUse:$s_y)]
>;


//===----------------------------------------------------------------------===//
// ONNXMulOp %X = ZHighUnstickOp (ZHighMulOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighMulPattern : Pat<
  (ZHighUnstickOp (ZHighMulOp (ZHighStickOp:$s_x $x, $_), (ZHighStickOp:$s_y $y, $_))),
  (ONNXMulOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXSubOp %X = ZHighUnstickOp (ZHighSubOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighSubPattern : Pat<
  (ZHighUnstickOp (ZHighSubOp (ZHighStickOp:$s_x $x, $_), (ZHighStickOp:$s_y $y, $_))),
  (ONNXSubOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXDivOp %X = ZHighUnstickOp (ZHighDivOp (ZHighStickOp
// %X),(ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighDivPattern : Pat<
  (ZHighUnstickOp (ZHighDivOp (ZHighStickOp:$s_x $x, $_), (ZHighStickOp:$s_y $y, $_))),
  (ONNXDivOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXLogOp %X = ZHighUnstickOp (ZHighLogOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceZHighLogPattern : Pat<
  (ZHighUnstickOp (ZHighLogOp (ZHighStickOp:$s_x $x, $_))),
  (ONNXLogOp $x),
  [(NotBlockArgument:$x), (HasOneUse:$s_x)]
>;

//===----------------------------------------------------------------------===//
// ONNXReluOp %X = ZHighUnstickOp (ZHighReluOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceZHighReluPattern : Pat<
  (ZHighUnstickOp (ZHighReluOp (ZHighStickOp:$s_x $x, $_))),
  (ONNXReluOp $x),
  [(NotBlockArgument:$x), (HasOneUse:$s_x)]
>;

#endif // ZHIGH_TO_ONNX
