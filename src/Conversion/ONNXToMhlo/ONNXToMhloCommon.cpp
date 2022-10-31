/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- ONNXToMhloCommon.cpp - ONNX dialects to Mhlo lowering ---------===//
//
// Copyright 2022
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the MHLO dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Support/TypeUtilities.hpp"
#include "stablehlo/dialect/BroadcastUtils.h"

using namespace mlir;

namespace onnx_mlir {

Value getShapedZero(
    Location loc, ConversionPatternRewriter &rewriter, Value &inp) {
  ShapedType inpType = inp.getType().cast<ShapedType>();
  Value broadcastedZero;
  if (inpType.hasStaticShape())
    broadcastedZero =
        rewriter.create<mhlo::ConstantOp>(loc, rewriter.getZeroAttr(inpType));
  else {
    Type elemType = inpType.getElementType();
    Value zero =
        rewriter.create<mhlo::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, inp);
    broadcastedZero = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
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
    Value broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(loc,
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
    Value broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(loc,
        broadcastedOutputType, operand, resultExtents,
        rewriter.getI64TensorAttr(broadcastDimensions));
    broadcastedOperands.push_back(broadcast);
  }
  return broadcastedOperands;
}

// This function satisfies the ArrayValueIndexCapture::DenseElementsAttr lambda
// type, using MHLO and ONNX operations.
DenseElementsAttr getDenseElementAttributeFromMhloValue(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto constantOp = dyn_cast_or_null<mhlo::ConstantOp>(definingOp)) {
    return constantOp.getValue().dyn_cast<DenseElementsAttr>();
  } else if (auto constantOp =
                 dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp)) {
    if (constantOp.value().has_value())
      return constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
  }
  return nullptr;
}

// This function satisfies the ArrayValueIndexCapture::LoadVal lambda
// type, using MHLO operations.
mlir::Value loadValuefromArrayAtIndexWithMhlo(mlir::OpBuilder &rewriter,
    mlir::Location loc, mlir::Value array, int64_t index) {
  Type type = array.getType();
  assert(isRankedShapedType(type) && "array must be ranked Shaped Type");
  Type elemType = getElementType(type);
  if (elemType.isa<IntegerType>()) {
    // cast to a tensor of index
    Type indexTensorType = RankedTensorType::get(
        onnx_mlir::getShape(type), rewriter.getIndexType());
    array = rewriter.create<arith::IndexCastOp>(loc, indexTensorType, array);
  } else if (!elemType.isa<IndexType>()) {
    llvm_unreachable("unsupported element type");
  }
  return rewriter.create<shape::GetExtentOp>(loc, array, index);
}

} // namespace onnx_mlir
