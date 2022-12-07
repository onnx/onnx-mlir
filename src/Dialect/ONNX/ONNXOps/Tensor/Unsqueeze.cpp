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

#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <typename OP_TYPE>
LogicalResult NewONNXCommonUnsqueezeOpShapeHelper<OP_TYPE>::customComputeShape(
    DimsExpr &unsqueezedDims) {
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());
  DimsExpr outputDims;
  Value data = operandAdaptor.data();
  int64_t dataRank = createIE->getTypeRank(data);

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
      outputDims.emplace_back(LiteralIndexExpr(1));
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
void NewONNXUnsqueezeOpShapeHelper::saveAxes() {
  // Create a ConstantOp associated with this Unsqueeze Op
  // There could be an issue if we were to generate a constant Op late in
  // lowering, but since we normalize them during the first shape inference, we
  // should never encounter a "saveAxles" situation during lowering.

#if 1
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(op);
  OnnxBuilder createONNX(builder, op->getLoc());
  Value constVal = createONNX.constantInt64(unsqueezedAxes);
  op->setOperand(1, constVal);
#else
  OpBuilder builder(op->getContext());
  auto tensorType =
      RankedTensorType::get(unsqueezedAxes.size(), builder.getI64Type());
  auto constDenseAttr =
      DenseElementsAttr::get(tensorType, llvm::makeArrayRef(unsqueezedAxes));
  builder.setInsertionPoint(op);
  auto constOp = builder.create<mlir::ONNXConstantOp>(
      op->getLoc(), mlir::Attribute(), constDenseAttr);
  op->setOperand(1, constOp.output());
#endif
}

template <>
void NewONNXUnsqueezeV11OpShapeHelper::saveAxes() {
  // Write attribues in op.
  OpBuilder builder(op->getContext());
  ArrayAttr newAxesAttr = builder.getI64ArrayAttr(unsqueezedAxes);
  ONNXUnsqueezeV11Op unsqueezeOp = llvm::cast<ONNXUnsqueezeV11Op>(op);
  unsqueezeOp.axesAttr(newAxesAttr);
}

template <>
LogicalResult NewONNXUnsqueezeOpShapeHelper::computeShape() {
  ONNXUnsqueezeOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  Value axes = operandAdaptor.axes();
  SmallVector<IndexExpr, 4> unsqueezedDims;
  createIE->getIntFromArrayAsSymbols(axes, unsqueezedDims);
  return customComputeShape(unsqueezedDims);
}

template <>
LogicalResult NewONNXUnsqueezeV11OpShapeHelper::computeShape() {
  ONNXUnsqueezeV11OpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  auto axesAttr = operandAdaptor.axesAttr();
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
    std::function<void(mlir::Region &)> doShapeInference) {
  auto dataType = data().getType().dyn_cast<RankedTensorType>();
  if (!dataType)
    return success();

  Type elementType = dataType.getElementType();
  NewONNXUnsqueezeOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

LogicalResult ONNXUnsqueezeV11Op::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  auto dataType = data().getType().dyn_cast<RankedTensorType>();
  if (!dataType)
    return success();

  Type elementType = dataType.getElementType();
  NewONNXUnsqueezeV11OpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct NewONNXCommonUnsqueezeOpShapeHelper<ONNXUnsqueezeOp>;
template struct NewONNXCommonUnsqueezeOpShapeHelper<ONNXUnsqueezeV11Op>;
} // namespace onnx_mlir
