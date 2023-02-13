// SPDX-License-Identifier: Apache-2.0

//===----- ShapeTransform.td - Pattern Match for ZHighShapeTransform ------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match optimizations for ZHigh using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ZHIGH_SHAPE_TRANSFORM_TD 
#define ZHIGH_SHAPE_TRANSFORM_TD 

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
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// DRR patterns 
//===----------------------------------------------------------------------===//

// Rewrite
// ```
//   zhigh.ShapeTransform {#map1} (zhigh.ShapeTransform {#map2} (%X))
// ```
// into
// ```
//   zhigh.ShapeTransform {#map} (%X)
// ```
// where `#map = compose(#map1, #map2)

def ShapeTransformComposePattern : Pat<
  (ZHighShapeTransformOp (ZHighShapeTransformOp $arg, $map1), $map2),
  (ZHighShapeTransformOp $arg, (GetComposedMap $map2, $map1)),
  []
>;

#endif // ZHIGH_SHAPE_TRANSFORM_TD
