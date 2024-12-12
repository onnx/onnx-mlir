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
def replaceZHighAddPattern : Pat<
  (ZHighUnstickOp (ZHighAddOp (ZHighStickOp:$s_x $x, $_, $_), (ZHighStickOp:$s_y $y, $_, $_))),
  (ONNXAddOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXMulOp %X = ZHighUnstickOp (ZHighMulOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighMulPattern : Pat<
  (ZHighUnstickOp (ZHighMulOp (ZHighStickOp:$s_x $x, $_, $_), (ZHighStickOp:$s_y $y, $_, $_))),
  (ONNXMulOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXSubOp %X = ZHighUnstickOp (ZHighSubOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighSubPattern : Pat<
  (ZHighUnstickOp (ZHighSubOp (ZHighStickOp:$s_x $x, $_, $_), (ZHighStickOp:$s_y $y, $_, $_))),
  (ONNXSubOp $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXDivOp %X = ZHighUnstickOp (ZHighDivOp (ZHighStickOp
// %X),(ZHighStickOp %Y))
// Note: turn off this pattern since NNPA is faster at this moment.
//===----------------------------------------------------------------------===//
// def replaceZHighDivPattern : Pat<
//   (ZHighUnstickOp (ZHighDivOp (ZHighStickOp:$s_x $x, $_, $_), (ZHighStickOp:$s_y $y, $_, $_))),
//   (ONNXDivOp $x, $y),
//   [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
// >;

//===----------------------------------------------------------------------===//
// ONNXMinOp %X = ZHighUnstickOp (ZHighMinOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighMinPattern : Pat<
  (ZHighUnstickOp:$u (ZHighMinOp (ZHighStickOp:$s_x $x, $_, $_), (ZHighStickOp:$s_y $y, $_, $_))),
  (CreateONNXMinOp $u, $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
>;

//===----------------------------------------------------------------------===//
// ONNXMaxOp %X = ZHighUnstickOp (ZHighMaxOp (ZHighStickOp %X),
// (ZHighStickOp %Y))
//===----------------------------------------------------------------------===//
def replaceZHighMaxPattern : Pat<
  (ZHighUnstickOp:$u (ZHighMaxOp (ZHighStickOp:$s_x $x, $_, $_), (ZHighStickOp:$s_y $y, $_, $_))),
  (CreateONNXMaxOp $u, $x, $y),
  [(NotBlockArgument:$x), (HasOneUse:$s_x), (NotBlockArgument:$y), (HasOneUse:$s_y)]
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
