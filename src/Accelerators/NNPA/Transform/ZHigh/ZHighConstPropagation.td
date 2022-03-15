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
include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.td"
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

// Useful test definitions.

// Check an ONNXConstantOp is using dense attribute or not.
def IsFromDenseONNXConstantOp:
    Constraint<CPred<"isFromDenseONNXConstantOp($_self)">,
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

// zhigh.Stick (c) = krnl.global(c1), where c1 is stickified data.
def ConstantStickPattern : Pat<
  (ZHighStickOp:$stickOp
     (ONNXConstantOp:$c $_, $_, $_, $_, $_, $_, $_, $_),
     $layout),
  (CreateConstantForStick $stickOp, $c, $layout),
  [(IsFromDenseONNXConstantOp:$c)]
>;

// zhigh.StickForLSTM (c1, c2, c3, c4) = krnl.global(c)
// where c is stickified data.
def ConstantStickForLSTMPattern : Pat<
  (ZHighStickForLSTMOp:$stickOp
     (ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_),
     (ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_),
     (ONNXConstantOp:$c3 $_, $_, $_, $_, $_, $_, $_, $_),
     (ONNXConstantOp:$c4 $_, $_, $_, $_, $_, $_, $_, $_)),
  (CreateConstantForStickForLSTM $stickOp, $c1, $c2, $c3, $c4),
  [(IsFromDenseONNXConstantOp:$c1), (IsFromDenseONNXConstantOp:$c2),
   (IsFromDenseONNXConstantOp:$c3), (IsFromDenseONNXConstantOp:$c4)]
>;

// zhigh.StickForGRU (c1, c2, c3) = krnl.global(c)
// where c is stickified data.
def ConstantStickForGRUPattern : Pat<
  (ZHighStickForGRUOp:$stickOp
     (ONNXConstantOp:$c1 $_, $_, $_, $_, $_, $_, $_, $_),
     (ONNXConstantOp:$c2 $_, $_, $_, $_, $_, $_, $_, $_),
     (ONNXConstantOp:$c3 $_, $_, $_, $_, $_, $_, $_, $_)),
  (CreateConstantForStickForGRU $stickOp, $c1, $c2, $c3),
  [(IsFromDenseONNXConstantOp:$c1), (IsFromDenseONNXConstantOp:$c2),
   (IsFromDenseONNXConstantOp:$c3)]
>;

#endif // ZHIGH_CONST_PROPAGATION
