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

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/Tensor/SqueezeUnsqueeze.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {

// Update axes attribute so that it contains only positive values.
void updateSqueezeOpNegativeAxis(ONNXSqueezeOp *op, ArrayRef<int64_t> axes) {
  updateNegativeAxis(op, axes);
}

void updateSqueezeOpNegativeAxis(ONNXSqueezeV11Op *op, ArrayRef<int64_t> axes) {
  updateNegativeAxisV11(op, axes);
}

template <typename Op, typename Adaptor, typename ShapeHelper>
LogicalResult ONNXSqueezeOpInferShapesCommon(Op *op,
    llvm::Optional<ArrayAttr> axisAttrs,
    std::function<void(mlir::Region &)> doShapeInference) {
  auto operandTy = op->data().getType().template cast<RankedTensorType>();
  auto elementType =
      op->data().getType().template cast<ShapedType>().getElementType();
  int64_t inRank = operandTy.getRank();

  SmallVector<int64_t, 4> axes;
  bool hasNegativeAxis = false;
  for (auto axisAttr : axisAttrs.value()) {
    int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
    if (axis < -inRank || axis >= inRank)
      return op->emitError("Invalid axis value");
    if (axis < 0) {
      axis = inRank + axis;
      hasNegativeAxis = true;
    }
    if (std::find(axes.begin(), axes.end(), axis) != axes.end())
      return op->emitError("Duplicated axes");
    axes.emplace_back(axis);
  }

  if (hasNegativeAxis) {
    updateSqueezeOpNegativeAxis(op, axes);
  }

  return shapeHelperInferShapes<ShapeHelper, Op, Adaptor>(*op, elementType);
}

// Helper function to return an ArrayAttr from an input shape
// All single dimensions will be returned
ArrayAttr getSqueezeOpAxesFromShape(
    OpBuilder builder, ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> axes;
  for (unsigned int i = 0; i < shape.size(); ++i) {
    if (shape[i] == 1) {
      axes.emplace_back(i);
    } else if (shape[i] == -1) {
      llvm_unreachable(
          "only static input shape currently supported with empty axes");
    }
  }
  return builder.getI64ArrayAttr(axes);
}

} // namespace

namespace onnx_mlir {

template <typename ShapeHelper, typename OperandAdaptor>
LogicalResult ONNXSqueezeOpShapeHelperCommon(ShapeHelper *shapeHelper,
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

  for (int i = 0; i < dataRank; ++i)
    if (std::find(axes.begin(), axes.end(), i) == axes.end())
      outputDims.emplace_back(dataBounds.getDim(i));

  // Save the final result.
  shapeHelper->setOutputDims(outputDims);

  return success();
}

LogicalResult ONNXSqueezeOpShapeHelper::computeShape(
    ONNXSqueezeOpAdaptor operandAdaptor) {
  auto axes = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (auto axesConstOp = getONNXConstantOp(axes)) {
    ArrayValueIndexCapture axesCapture(axes, fGetDenseVal, fLoadVal);
    axesCapture.getSymbolList(indexExprArray);
  } else if (!axes.getType().template isa<NoneType>()) {
    llvm_unreachable("dynamic axes not yet supported");
  }

  return ONNXSqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

LogicalResult ONNXSqueezeV11OpShapeHelper::computeShape(
    ONNXSqueezeV11OpAdaptor operandAdaptor) {
  auto axesAttr = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (axesAttr.has_value()) {
    ArrayAttributeIndexCapture axesCapture(axesAttr.value());
    auto axesRank = axesCapture.size();
    for (unsigned i = 0; i < axesRank; ++i) {
      indexExprArray.emplace_back(axesCapture.getLiteral(i));
    }
  }
  return ONNXSqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXSqueezeOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto dataType = data().getType().dyn_cast<RankedTensorType>();
  if (!dataType)
    return success();

  OpBuilder builder(getContext());
  llvm::Optional<ArrayAttr> optionalAttr;

  if (isFromNone(axes())) {
    auto axesAttr = getSqueezeOpAxesFromShape(builder, dataType.getShape());
    optionalAttr.emplace(axesAttr);

    // Create a ConstantOp associated with this Squeeze Op
    auto tensorType =
        RankedTensorType::get(ArrayAttrSize(axesAttr), builder.getI64Type());
    SmallVector<int64_t, 4> values;
    for (auto attr : axesAttr.getValue()) {
      values.emplace_back(attr.cast<IntegerAttr>().getInt());
    }
    auto constDenseAttr =
        DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
    builder.setInsertionPoint(*this);
    auto constOp = builder.create<mlir::ONNXConstantOp>(
        getLoc(), mlir::Attribute(), constDenseAttr);
    mlir::Value constRes = constOp.output();
    setOperand(1, constRes);
  } else if (auto axesConstOp = getONNXConstantOp(axes())) {
    auto axesAttr = createArrayAttrFromConstantOp(builder, axesConstOp);
    optionalAttr.emplace(axesAttr);
  } else {
    llvm_unreachable("dynamic axes not yet supported");
  }

  return ONNXSqueezeOpInferShapesCommon<ONNXSqueezeOp, ONNXSqueezeOpAdaptor,
      ONNXSqueezeOpShapeHelper>(this, optionalAttr, doShapeInference);
}

LogicalResult ONNXSqueezeV11Op::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto dataType = data().getType().dyn_cast<RankedTensorType>();
  if (!dataType)
    return success();

  if (!axes()) {
    OpBuilder builder(getContext());

    auto newAxesAttr = getSqueezeOpAxesFromShape(builder, dataType.getShape());

    // Update the axes attribute
    axesAttr(newAxesAttr);
  }

  return ONNXSqueezeOpInferShapesCommon<ONNXSqueezeV11Op,
      ONNXSqueezeV11OpAdaptor, ONNXSqueezeV11OpShapeHelper>(
      this, axes(), doShapeInference);
}
