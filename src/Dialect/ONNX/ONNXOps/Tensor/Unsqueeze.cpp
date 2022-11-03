/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ SqueezeUnsqueeze.cpp - ONNX Operations ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Squeeze and Unsqueeze
// operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/Tensor/SqueezeUnsqueeze.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {

void updateUnsqueezeOpNegativeAxis(
    ONNXUnsqueezeOp *op, ArrayRef<int64_t> axes) {
  updateNegativeAxis(op, axes);
}

void updateUnsqueezeOpNegativeAxis(
    ONNXUnsqueezeV11Op *op, ArrayRef<int64_t> axes) {
  updateNegativeAxisV11(op, axes);
}

template <typename Op, typename Adaptor, typename ShapeHelper>
LogicalResult ONNXUnsqueezeOpInferShapesCommon(Op *op,
    llvm::Optional<ArrayAttr> axisAttrs,
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!op->data().getType().template isa<RankedTensorType>())
    return success();

  auto operandTy = op->data().getType().template cast<RankedTensorType>();
  auto elementType =
      op->data().getType().template cast<ShapedType>().getElementType();
  int64_t inRank = operandTy.getRank();

  if (!axisAttrs)
    return op->emitError("Axes attribute is required");

  SmallVector<int64_t, 4> axes;
  bool hasNegativeAxis = false;
  int64_t outRank = inRank + axisAttrs.value().size();
  for (auto axisAttr : axisAttrs.value()) {
    int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
    if (axis < -outRank || axis >= outRank)
      return op->emitError("Invalid axis value");
    if (axis < 0) {
      axis = outRank + axis;
      hasNegativeAxis = true;
    }
    if (std::find(axes.begin(), axes.end(), axis) == axes.end())
      axes.emplace_back(axis);
    else
      return op->emitError("Duplicated axes");
  }

  if (hasNegativeAxis) {
    updateUnsqueezeOpNegativeAxis(op, axes);
  }

  return shapeHelperInferShapes<ShapeHelper, Op, Adaptor>(*op, elementType);
}

} // namespace

namespace onnx_mlir {

template <typename ShapeHelper, typename OperandAdaptor>
LogicalResult ONNXUnsqueezeOpShapeHelperCommon(ShapeHelper *shapeHelper,
    OperandAdaptor operandAdaptor, ArrayRef<IndexExpr> indexExprArray) {
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get axis values. They are expected to be normalized before so that there
  // is no negative values.
  SmallVector<int64_t, 4> axes;
  for (auto axisAttr : indexExprArray) {
    int64_t axis = axisAttr.getLiteral();
    assert(axis >= 0 && "Invalid axis");
    axes.emplace_back(axis);
  }

  int64_t outRank = dataRank + axes.size();
  for (int i = 0, j = 0; i < outRank || j < dataRank; ++i)
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      outputDims.emplace_back(LiteralIndexExpr(1));
    else
      outputDims.emplace_back(dataBounds.getDim(j++));

  // Save the final result.
  shapeHelper->setOutputDims(outputDims);

  return success();
}

LogicalResult ONNXUnsqueezeOpShapeHelper::computeShape(
    ONNXUnsqueezeOpAdaptor operandAdaptor) {
  auto axes = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (auto axesConstOp = getONNXConstantOp(axes)) {
    ArrayValueIndexCapture axesCapture(axes, fGetDenseVal, fLoadVal);
    axesCapture.getSymbolList(indexExprArray);
  } else if (!axes.getType().template isa<NoneType>()) {
    llvm_unreachable("dynamic axes not yet supported");
  }

  return ONNXUnsqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

LogicalResult ONNXUnsqueezeV11OpShapeHelper::computeShape(
    ONNXUnsqueezeV11OpAdaptor operandAdaptor) {
  auto axesAttr = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  ArrayAttributeIndexCapture axesCapture(axesAttr);
  auto axesRank = axesCapture.size();
  for (unsigned i = 0; i < axesRank; ++i) {
    indexExprArray.emplace_back(axesCapture.getLiteral(i));
  }
  return ONNXUnsqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXUnsqueezeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto builder = mlir::Builder(getContext());
  llvm::Optional<ArrayAttr> optionalAttr;
  if (auto axesConstOp = getONNXConstantOp(axes())) {
    auto axesAttr = createArrayAttrFromConstantOp(builder, axesConstOp);
    optionalAttr.emplace(axesAttr);
  } else if (!axes().getType().isa<NoneType>()) {
    // Cannot handle Non-constant axes
    // Hope further transformation may creat constant axes
    return success();
  }
  return ONNXUnsqueezeOpInferShapesCommon<ONNXUnsqueezeOp,
      ONNXUnsqueezeOpAdaptor, ONNXUnsqueezeOpShapeHelper>(
      this, optionalAttr, doShapeInference);
}

LogicalResult ONNXUnsqueezeV11Op::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  return ONNXUnsqueezeOpInferShapesCommon<ONNXUnsqueezeV11Op,
      ONNXUnsqueezeV11OpAdaptor, ONNXUnsqueezeV11OpShapeHelper>(
      this, axes(), doShapeInference);
}
