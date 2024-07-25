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
def CreateONNXMinOp : NativeCodeCall<"$_builder.create<ONNXMinOp>($_loc, $0.getType(), ValueRange({$1, $2}))">;
def CreateONNXMaxOp : NativeCodeCall<"$_builder.create<ONNXMaxOp>($_loc, $0.getType(), ValueRange({$1, $2}))">;

//===----------------------------------------------------------------------===//
// ONNXAddOp %X = ZHighUnstickOp (ZHighAddOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighAddPattern1 : Pat<
  (ZHighUnstickOp (ZHighAddOp (ZHighStickOp:$s_x $x, $_, $_), $y)),
  (ONNXAddOp $x, (ZHighUnstickOp $y)),
  [(NotBlockArgument:$x), (HasOneUse:$s_x)]
>;

def replaceZHighAddPattern2 : Pat<
  (ZHighUnstickOp (ZHighAddOp $x, (ZHighStickOp:$s_y $y, $_, $_))),
  (ONNXAddOp (ZHighUnstickOp $x), $y),
  [(NotBlockArgument:$y), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXMulOp %X = ZHighUnstickOp (ZHighMulOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighMulPattern1 : Pat<
  (ZHighUnstickOp (ZHighMulOp (ZHighStickOp:$s_x $x, $_, $_), $y)),
  (ONNXMulOp $x, (ZHighUnstickOp $y)),
  [(NotBlockArgument:$x), (HasOneUse:$s_x)], [ ],
  (addBenefit 1)
>;

def replaceZHighMulPattern2 : Pat<
  (ZHighUnstickOp (ZHighMulOp $x, (ZHighStickOp:$s_y $y, $_, $_))),
  (ONNXMulOp (ZHighUnstickOp $x), $y),
  [(NotBlockArgument:$y), (HasOneUse:$s_y)], [],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// ONNXSubOp %X = ZHighUnstickOp (ZHighSubOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighSubPattern1 : Pat<
  (ZHighUnstickOp (ZHighSubOp (ZHighStickOp:$s_x $x, $_, $_), $y)),
  (ONNXSubOp $x, (ZHighUnstickOp $y)),
  [(NotBlockArgument:$x), (HasOneUse:$s_x)], [ ],
  (addBenefit 1)
>;

def replaceZHighSubPattern2 : Pat<
  (ZHighUnstickOp (ZHighSubOp $x, (ZHighStickOp:$s_y $y, $_, $_))),
  (ONNXSubOp (ZHighUnstickOp $x), $y),
  [(NotBlockArgument:$y), (HasOneUse:$s_y)], [ ],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// ONNXDivOp %X = ZHighUnstickOp (ZHighDivOp (ZHighStickOp
// %X),(ZHighStickOp %Y))
// Note: turn off this pattern since NNPA is faster at this moment.
//===----------------------------------------------------------------------===//
//def replaceZHighDivPattern1 : Pat<
//  (ZHighUnstickOp (ZHighDivOp (ZHighStickOp:$s_x $x, $_), $y)),
//  (ONNXDivOp $x, (ZHighUnstickOp $y)),
//  [(NotBlockArgument:$x), (HasOneUse:$s_x)], [ ],
//  (addBenefit 1)
//>;
//
//def replaceZHighDivPattern2 : Pat<
//  (ZHighUnstickOp (ZHighDivOp $x, (ZHighStickOp:$s_y $y, $_))),
//  (ONNXDivOp (ZHighUnstickOp $x), $y),
//  [(NotBlockArgument:$y), (HasOneUse:$s_y)], [ ],
//  (addBenefit 0)
//>;

//===----------------------------------------------------------------------===//
// ONNXMinOp %X = ZHighUnstickOp (ZHighMinOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighMinPattern1 : Pat<
  (ZHighUnstickOp:$u (ZHighMinOp (ZHighStickOp:$s_x $x, $_, $_), $y)),
  (CreateONNXMinOp $u, $x, (ZHighUnstickOp $y)),
  [(NotBlockArgument:$x), (HasOneUse:$s_x)], [ ],
  (addBenefit 1)
>;

def replaceZHighMinPattern2 : Pat<
  (ZHighUnstickOp:$u (ZHighMinOp $x, (ZHighStickOp:$s_y $y, $_, $_))),
  (CreateONNXMinOp $u, (ZHighUnstickOp $x), $y),
  [(NotBlockArgument:$y), (HasOneUse:$s_y)], [ ],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// ONNXMaxOp %X = ZHighUnstickOp (ZHighMaxOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighMaxPattern1 : Pat<
  (ZHighUnstickOp:$u (ZHighMaxOp (ZHighStickOp:$s_x $x, $_, $_), $y)),
  (CreateONNXMaxOp $u, $x, (ZHighUnstickOp $y)),
  [(NotBlockArgument:$x), (HasOneUse:$s_x)], [ ],
  (addBenefit 1)
>;

def replaceZHighMaxPattern2 : Pat<
  (ZHighUnstickOp:$u (ZHighMaxOp $x, (ZHighStickOp:$s_y $y, $_, $_))),
  (CreateONNXMaxOp $u, (ZHighUnstickOp $x), $y),
  [(NotBlockArgument:$y), (HasOneUse:$s_y)], [ ],
  (addBenefit 0)
>;

//===----------------------------------------------------------------------===//
// ONNXReluOp %X = ZHighUnstickOp (ZHighReluOp (ZHighStickOp %X))
//===----------------------------------------------------------------------===//
def replaceZHighReluPattern : Pat<
  (ZHighUnstickOp (ZHighReluOp (ZHighStickOp:$s_x $x, $_, $_))),
  (ONNXReluOp $x),
  [(NotBlockArgument:$x), (HasOneUse:$s_x)]
>;

//===----------------------------------------------------------------------===//
// ONNXLogOp %X = ZHighUnstickOp (ZHighLogOp (ZHighStickOp %X))
// Note: turn off this pattern since NNPA is faster at this moment.
//===----------------------------------------------------------------------===//
// def replaceZHighLogPattern : Pat<
//   (ZHighUnstickOp (ZHighLogOp (ZHighStickOp:$s_x $x, $_))),
//   (ONNXLogOp $x),
//   [(NotBlockArgument:$x), (HasOneUse:$s_x)]
// >;

//===----------------------------------------------------------------------===//
// ONNXExpOp %X = ZHighUnstickOp (ZHighExpOp (ZHighStickOp %X))
// Note: turn off this pattern since NNPA is faster at this moment.
//===----------------------------------------------------------------------===//
// def replaceZHighExpPattern : Pat<
//   (ZHighUnstickOp (ZHighExpOp (ZHighStickOp:$s_x $x, $_))),
//   (ONNXExpOp $x),
//   [(NotBlockArgument:$x), (HasOneUse:$s_x)]
// >;

//===----------------------------------------------------------------------===//
// ONNXTanhOp %X = ZHighUnstickOp (ZHighTanhOp (ZHighStickOp %X))
// Note: turn off this pattern since NNPA is faster at this moment.
//===----------------------------------------------------------------------===//
// def replaceZHighTanhPattern : Pat<
//   (ZHighUnstickOp (ZHighTanhOp (ZHighStickOp:$s_x $x, $_))),
//   (ONNXTanhOp $x),
//   [(NotBlockArgument:$x), (HasOneUse:$s_x)]
// >;

//===----------------------------------------------------------------------===//
// ONNXSigmoidOp %X = ZHighUnstickOp (ZHighSigmoidOp (ZHighStickOp %X))
// Note: turn off this pattern since NNPA is faster at this moment.
//===----------------------------------------------------------------------===//
// def replaceZHighSigmoidPattern : Pat<
//   (ZHighUnstickOp (ZHighSigmoidOp (ZHighStickOp:$s_x $x, $_))),
//   (ONNXSigmoidOp $x),
//   [(NotBlockArgument:$x), (HasOneUse:$s_x)]
// >;

#endif // ZHIGH_TO_ONNX
