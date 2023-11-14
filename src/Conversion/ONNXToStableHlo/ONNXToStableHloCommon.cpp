/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- ONNXToStableHloCommon.cpp - ONNX dialects to StableHlo lowering
//---------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the StableHlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"
#include "stablehlo/dialect/BroadcastUtils.h"

using namespace mlir;

namespace onnx_mlir {

Value getShapedZero(
    Location loc, ConversionPatternRewriter &rewriter, Value &inp) {
  ShapedType inpType = inp.getType().cast<ShapedType>();
  Value broadcastedZero;
  if (inpType.hasStaticShape())
    broadcastedZero = rewriter.create<stablehlo::ConstantOp>(
        loc, rewriter.getZeroAttr(inpType));
  else {
    Type elemType = inpType.getElementType();
    Value zero = rewriter.create<stablehlo::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, inp);
    broadcastedZero = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
        loc, inpType, zero, shape, rewriter.getI64TensorAttr({}));
  }
  return broadcastedZero;
}

llvm::SmallVector<Value, 4> getBroadcastedOperands(Operation *op,
    ConversionPatternRewriter &rewriter, Location loc, int64_t outputRank) {
  llvm::SmallVector<Value, 4> broadcastedOperands;
  Type outputType = *op->result_type_begin();
  assert(outputType.isa<ShapedType>() && "output type is not shaped");
  ShapedType outputShapedType = outputType.cast<ShapedType>();
  Type elementType =
      op->getOperands()[0].getType().dyn_cast<ShapedType>().getElementType();
  RankedTensorType broadcastedOutputType =
      RankedTensorType::get(outputShapedType.getShape(), elementType);

  Value resultExtents =
      mlir::hlo::computeNaryElementwiseBroadcastingResultExtents(
          loc, op->getOperands(), rewriter);
  for (Value operand : op->getOperands()) {
    RankedTensorType operandType =
        operand.getType().dyn_cast<RankedTensorType>();
    assert(operandType != nullptr && "operand type is not ranked");
    SmallVector<int64_t, 4> broadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(outputRank - operandType.getRank(), outputRank));
    Value broadcast = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc,
        broadcastedOutputType, operand, resultExtents,
        rewriter.getI64TensorAttr(broadcastDimensions));
    broadcastedOperands.push_back(broadcast);
  }
  return broadcastedOperands;
}

llvm::SmallVector<Value, 4> getBroadcastedOperands(
    llvm::SmallVector<Value, 4> &operands, Type outputType,
    ConversionPatternRewriter &rewriter, Location loc, int64_t outputRank) {
  llvm::SmallVector<Value, 4> broadcastedOperands;
  assert(outputType.isa<ShapedType>() && "output type is not shaped");
  ShapedType outputShapedType = outputType.cast<ShapedType>();
  Type elementType =
      operands[0].getType().dyn_cast<ShapedType>().getElementType();
  RankedTensorType broadcastedOutputType =
      RankedTensorType::get(outputShapedType.getShape(), elementType);

  Value resultExtents =
      mlir::hlo::computeNaryElementwiseBroadcastingResultExtents(
          loc, operands, rewriter);
  for (Value operand : operands) {
    RankedTensorType operandType =
        operand.getType().dyn_cast<RankedTensorType>();
    assert(operandType != nullptr && "operand type is not ranked");
    SmallVector<int64_t, 4> broadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(outputRank - operandType.getRank(), outputRank));
    Value broadcast = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc,
        broadcastedOutputType, operand, resultExtents,
        rewriter.getI64TensorAttr(broadcastDimensions));
    broadcastedOperands.push_back(broadcast);
  }
  return broadcastedOperands;
}

ElementsAttr getElementAttributeFromStablehloValue(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto constantOp = dyn_cast_or_null<stablehlo::ConstantOp>(definingOp))
    return constantOp.getValue().dyn_cast<ElementsAttr>();
  else if (auto constantOp =
               dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp)) {
    if (constantOp.getValue().has_value())
      return constantOp.getValueAttr().dyn_cast<ElementsAttr>();
  }
  return nullptr;
}

} // namespace onnx_mlir
