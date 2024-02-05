// SPDX-License-Identifier: Apache-2.0

//===------- ZHighUnstick.td - Pattern Match for ZHighUnstick -------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef UNSTICK_TD 
#define UNSTICK_TD 

#ifndef OP_BASE
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td"
include "src/Dialect/ONNX/ONNX.td"
#endif // OP_BASE

include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// DRR patterns 
//===----------------------------------------------------------------------===//

// zhigh.Unstick (zhigh.Stick (%X)) = %X
def UnstickStickRemovalPattern : Pat<
  (ZHighUnstickOp (ZHighStickOp $arg, $_)),
  (replaceWithValue $arg)
>;

#endif // UNSTICK_TD
