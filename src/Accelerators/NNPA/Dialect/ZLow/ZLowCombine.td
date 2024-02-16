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
include "mlir/IR/PatternBase.td"
include "ZLow.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    list<dag> supplementalPatterns = [],
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

// Remove pairs of dlf16_to_f32 and f32_to_dlf16
def DLF16ConversionOpPattern: Pat<
  (ZLowConvertF32ToDLF16Op (ZLowConvertDLF16ToF32Op $arg)),
  (replaceWithValue $arg)
>;

#endif // ZLOW_COMBINE
