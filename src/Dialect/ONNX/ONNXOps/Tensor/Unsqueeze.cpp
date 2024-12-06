/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ UnsqueezeUnsqueeze.cpp - ONNX Operations
//------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Unsqueeze and Unsqueeze
// operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <typename OP_TYPE>
LogicalResult ONNXCommonUnsqueezeOpShapeHelper<OP_TYPE>::customComputeShape(
    DimsExpr &unsqueezedDims) {
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr outputDims;
  Value data = operandAdaptor.getData();
  if (!hasShapeAndRank(data)) {
    return failure();
  }
  int64_t dataRank = createIE->getShapedTypeRank(data);

  // Init state.
  bool modified = false;
  unsqueezedAxes.clear();
  int64_t outRank = dataRank + unsqueezedDims.size();
  // Normalize the axis values, record modified values in squeezedDims.
  for (uint64_t i = 0; i < unsqueezedDims.size(); ++i) {
    // Check if the dimension to squeeze is a literal and in range.
    if (!unsqueezedDims[i].isLiteral())
      return op->emitError(
          "Can not unsqueeze from dynamic dimensions at this time");
    int64_t a = unsqueezedDims[i].getLiteral();
    if (a < -outRank || a >= outRank)
      return op->emitError("Invalid axis value");
    // Handle negative axis.
    if (a < 0) {
      a += outRank;
      modified = true;
    }
    if (std::find(unsqueezedAxes.begin(), unsqueezedAxes.end(), a) ==
        unsqueezedAxes.end())
      unsqueezedAxes.emplace_back(a);
    else
      return op->emitError("Duplicated axes");
  }
  // Now compute the output dims
  for (int64_t i = 0, j = 0; i < outRank || j < dataRank; ++i)
    if (std::find(unsqueezedAxes.begin(), unsqueezedAxes.end(), i) !=
        unsqueezedAxes.end())
      // found i in unsqueeze axles.
      outputDims.emplace_back(LitIE(1));
    else
      outputDims.emplace_back(createIE->getShapeAsDim(data, j++));

  // Save the final result.
  setOutputDims(outputDims);
  // Save the modified state
  if (modified)
    saveAxes();
  return success();
}

template <>
void ONNXUnsqueezeOpShapeHelper::saveAxes() {
  // Create a ConstantOp associated with this Unsqueeze Op
  // There could be an issue if we were to generate a constant Op late in
  // lowering, but since we normalize them during the first shape inference, we
  // should never encounter a "saveAxles" situation during lowering.

  ONNXUnsqueezeOp unsqueezeOp = llvm::cast<ONNXUnsqueezeOp>(op);
  SaveOnnxConstInOp(op, unsqueezeOp.getAxesMutable(), unsqueezedAxes);
}

template <>
void ONNXUnsqueezeV11OpShapeHelper::saveAxes() {
  SaveOnnxAttrInOp<ONNXUnsqueezeV11Op>(op, unsqueezedAxes,
      [](ONNXUnsqueezeV11Op op, ArrayAttr attr) { op.setAxesAttr(attr); });
}

template <>
LogicalResult ONNXUnsqueezeOpShapeHelper::computeShape() {
  ONNXUnsqueezeOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  Value axes = operandAdaptor.getAxes();
  SmallVector<IndexExpr, 4> unsqueezedDims;
  createIE->getIntFromArrayAsSymbols(axes, unsqueezedDims);
  return customComputeShape(unsqueezedDims);
}

template <>
LogicalResult ONNXUnsqueezeV11OpShapeHelper::computeShape() {
  ONNXUnsqueezeV11OpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  auto axesAttr = operandAdaptor.getAxesAttr();
  assert(axesAttr && "expected axes attribute");
  SmallVector<IndexExpr, 4> unsqueezedDims;
  createIE->getIntFromArrayAsLiterals(axesAttr, unsqueezedDims);
  return customComputeShape(unsqueezedDims);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXUnsqueezeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto dataType = mlir::dyn_cast<RankedTensorType>(getData().getType());
  if (!dataType)
    return success();

  Type elementType = dataType.getElementType();
  ONNXUnsqueezeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

LogicalResult ONNXUnsqueezeV11Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  auto dataType = mlir::dyn_cast<RankedTensorType>(getData().getType());
  if (!dataType)
    return success();

  Type elementType = dataType.getElementType();
  ONNXUnsqueezeV11OpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXCommonUnsqueezeOpShapeHelper<ONNXUnsqueezeOp>;
template struct ONNXCommonUnsqueezeOpShapeHelper<ONNXUnsqueezeV11Op>;
} // namespace onnx_mlir
