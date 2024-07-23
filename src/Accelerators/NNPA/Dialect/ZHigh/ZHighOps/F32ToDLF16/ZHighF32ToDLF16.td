// SPDX-License-Identifier: Apache-2.0

//===------- ZHighF32ToDLF16.td - Pattern Match for ZHighUnstick ----------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef F32_TO_DLF16_TD 
#define F32_TO_DLF16_TD 

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

// zhigh.F32ToDLF16 (zhigh.DLF16ToF32(%X)) = %X
def ConversionRemovalPattern : Pat<
  (ZHighF32ToDLF16Op (ZHighDLF16ToF32Op $arg), $saturation),
  (replaceWithValue $arg)
>;

#endif // F32_TO_DLF16_TD
