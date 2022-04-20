// SPDX-License-Identifier: Apache-2.0

//===--------- ZLowCombine.td - Pattern Match for ZLowDialect -------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZLow using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ZLOW_COMBINE
#define ZLOW_COMBINE

#ifndef OP_BASE
include "ZLow.td"
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

// zlow.dummy(X) is replaced by X.
def RemoveDummyOpPattern: Pat<
  (ZLowDummyOp $arg),
  (replaceWithValue $arg)
>;

#endif // ZLOW_COMBINE
