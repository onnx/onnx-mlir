// SPDX-License-Identifier: Apache-2.0

//===------- ZHighDecomposeStickUnstick.td - Pattern Match ----------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef DECOMPOSE_STICK_UNSTICK_TD 
#define DECOMPOSE_STICK_UNSTICK_TD 

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

def NoneLayoutAttr: NativeCodeCall<"StringAttr()">;

// zhigh.Stick $input, $layout = onnx.LayoutTransform (zhigh.F32ToDLF16 $input), $layout
// Not support NHWC layout since the stick does transpose internally. We can
// introduce transpose explicitly, but need to evaluate its overhead.
def DecomposeStickPattern: Pat<
  (ZHighStickOp $input, $layout, $saturation),
  (ONNXLayoutTransformOp (ZHighF32ToDLF16Op $input, $saturation), $layout),
  [(IsNotNHWCLayout $layout)]
>;

// zhigh.Unstick $input = zhigh.F32ToDLF16 (onnx.LayoutTransform $input, $none_layout)
// Not support NHWC layout since the unstick does transpose internally. We can
// introduce transpose explicitly, but need to evaluate its overhead.
def DecomposeUnstickPattern: Pat<
  (ZHighUnstickOp $input),
  (ZHighDLF16ToF32Op (ONNXLayoutTransformOp $input, (NoneLayoutAttr))),
  [(TensorHasNoNHWCLayout $input)]
>;

#endif // DECOMPOSE_STICK_UNSTICK_TD
