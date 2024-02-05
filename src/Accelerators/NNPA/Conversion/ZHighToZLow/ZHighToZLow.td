// SPDX-License-Identifier: Apache-2.0

//===--- ZHighToZLow.td - Replacing ZHigh Ops by ZLow Ops -*- tablegen ----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ZHIGH_TO_ZLOW
#define ZHIGH_TO_ZLOW

#ifndef OP_BASE
include "Dialect/ZHigh/ZHigh.td"
include "Dialect/ZLow/ZLow.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;


#endif // ZHIGH_TO_ZLOW 
