// SPDX-License-Identifier: Apache-2.0

//===---- ZHighConstPropagation.td - Pattern Match for ZHighDialect -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ZHIGH_CONST_PROPAGATION
#define ZHIGH_CONST_PROPAGATION

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

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// Useful test definitions.

// Check an ONNXConstantOp is using dense attribute or not.
def IsFromDenseONNXConstantOp:
    Constraint<CPred<"isDenseONNXConstant($_self)">,
  "Value is produced by a dense ONNXConstantOp">;

// Constant propagation for stickify
def CreateConstantForStick: NativeCodeCall<
  "createConstantForStick($_builder, $0, $1, $2)"
>;

def CreateConstantForStickForLSTM : NativeCodeCall<
  "createConstantForStickForLSTM($_builder, $0, $1, $2, $3, $4)"
>;

def CreateConstantForStickForGRU : NativeCodeCall<
  "createConstantForStickForGRU($_builder, $0, $1, $2, $3)"
>;

#endif // ZHIGH_CONST_PROPAGATION
