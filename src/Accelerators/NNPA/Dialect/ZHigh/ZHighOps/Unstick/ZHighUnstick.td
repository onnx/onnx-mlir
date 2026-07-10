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
  (ZHighUnstickOp (ZHighStickOp $arg, $_, $_)),
  (replaceWithValue $arg)
>;

// onnx.Dim (zhigh.Unstick (%X)) = onnx.Dim %X
// For NHWC layout.
def DimUnstickNHWCRemovalPattern : Pat<
  (ONNXDimOp (ZHighUnstickOp $arg), $axis),
  (ONNXDimOp $arg, (GetAxisNHWC $axis)),
  [(TensorHasNHWCLayout $arg), (HasNNPAUse $arg)]
>;

// onnx.Dim (zhigh.Unstick (%X)) = onnx.Dim %X
// For non-NHWC layout.
def DimUnstickRemovalPattern : Pat<
  (ONNXDimOp (ZHighUnstickOp $arg), $axis),
  (ONNXDimOp $arg, $axis),
  [(TensorHasNoNHWCLayout $arg), (HasNNPAUse $arg)]
>;

// Create a DenseElementsAttr from a float attribute and an element type.
def createDenseElementsAttrFromFloatAttr : NativeCodeCall<
  "onnx_mlir::createDenseElementsAttrFromFloatAttr($_builder, mlir::cast<ShapedType>($0.getType()).getElementType(), $1)">;

// Create a DenseElementsAttr with full static shape from unstick output type.
def createDenseElementsAttrFromFloatAttrWithShape : NativeCodeCall<
  "DenseElementsAttr::get(mlir::cast<ShapedType>($0.getType()), $1)">;

// zhigh.Unstick (zhigh.StickifiedConstantOfShape $shape, $value_attr, $layout) with static shape
// Generate ONNXConstantOp directly without ONNXExpandOp.
def UnstickConstantOfShapeStaticPattern: Pat<
  (ZHighUnstickOp:$unstick (ZHighStickifiedConstantOfShapeOp:$stick $shape, $value_attr, $layout)),
  (ONNXConstantOpFromDenseAttr (createDenseElementsAttrFromFloatAttrWithShape $unstick, $value_attr)),
  [(IsStaticShapeTensor $unstick)],
  [],
  (addBenefit 1)
>;

// zhigh.Unstick (zhigh.StickifiedConstantOfShape $shape, $value_attr, $layout) with dynamic shape
// Generate ONNXExpandOp with scalar constant.
def UnstickConstantOfShapePattern: Pat<
  (ZHighUnstickOp:$unstick (ZHighStickifiedConstantOfShapeOp:$stick $shape, $value_attr, $layout)),
  (ONNXExpandOp (ONNXConstantOpFromDenseAttr (createDenseElementsAttrFromFloatAttr $unstick, $value_attr)), $shape),
  [],
  [],
  (addBenefit 0)
>;

#endif // UNSTICK_TD
