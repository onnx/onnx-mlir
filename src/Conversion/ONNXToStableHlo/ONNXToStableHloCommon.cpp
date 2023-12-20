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
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

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
  Value resultExtents =
      mlir::hlo::computeNaryElementwiseBroadcastingResultExtents(
          loc, op->getOperands(), rewriter);
  for (Value operand : op->getOperands()) {
    RankedTensorType operandType =
        operand.getType().dyn_cast<RankedTensorType>();
    assert(operandType != nullptr && "operand type is not ranked");
    SmallVector<int64_t, 4> broadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(outputRank - operandType.getRank(), outputRank));
    Type elementType =
        operand.getType().dyn_cast<ShapedType>().getElementType();
    RankedTensorType broadcastedOutputType =
        RankedTensorType::get(outputShapedType.getShape(), elementType);
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
  Value resultExtents =
      mlir::hlo::computeNaryElementwiseBroadcastingResultExtents(
          loc, operands, rewriter);
  for (Value operand : operands) {
    RankedTensorType operandType =
        operand.getType().dyn_cast<RankedTensorType>();
    assert(operandType != nullptr && "operand type is not ranked");
    SmallVector<int64_t, 4> broadcastDimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(outputRank - operandType.getRank(), outputRank));
    Type elementType =
        operands[0].getType().dyn_cast<ShapedType>().getElementType();
    RankedTensorType broadcastedOutputType =
        RankedTensorType::get(outputShapedType.getShape(), elementType);
    Value broadcast = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(loc,
        broadcastedOutputType, operand, resultExtents,
        rewriter.getI64TensorAttr(broadcastDimensions));
    broadcastedOperands.push_back(broadcast);
  }
  return broadcastedOperands;
}

ElementsAttr getElementAttributeFromConstValue(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto constantOp = dyn_cast_or_null<stablehlo::ConstantOp>(definingOp)) {
    return constantOp.getValue().dyn_cast<ElementsAttr>();
  } else if (auto constantOp =
                 dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp)) {
    if (constantOp.getValue().has_value())
      return constantOp.getValueAttr().dyn_cast<ElementsAttr>();
  }
  return nullptr;
}

DenseIntElementsAttr GetI64ElementsAttr(
    ArrayRef<int64_t> values, Builder *builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

namespace {
// Returns the DenseElementsAttr of input if it's a stablehlo constant or
// onnx.Constant. Otherwise returns a nullptr attribute.
DenseElementsAttr getDenseElementAttrFromConstValue(mlir::Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (auto globalOp = dyn_cast_or_null<stablehlo::ConstantOp>(definingOp)) {
    return globalOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  } else if (auto constOp =
                 dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp)) {
    if (constOp.getValue().has_value())
      return constOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  }
  return nullptr;
}
} // namespace

// Emit an ONNXSqueezeOp. If the input is constant, do const propagation,
/// and return a constant.
Value foldOrEmitONNXSqueezeOpStableHlo(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  TensorType tensorType = create.onnx.toTensor(resultType);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    DenseElementsAttr squeezedElements = inputElements.reshape(tensorType);
    Value constVal = create.onnx.constant(squeezedElements);
    return constVal;
  } else {
    return rewriter
        .create<ONNXSqueezeOp>(loc, tensorType, create.onnx.toTensor(input),
            create.onnx.constantInt64({axis}))
        .getResult();
  }
}

/// Emit an ONNXUnsqueezeOp. If the input is constant, do const
/// propagation, and return a constant.
Value foldOrEmitONNXUnsqueezeOpStableHlo(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  TensorType tensorType = create.onnx.toTensor(resultType);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    DenseElementsAttr unsqueezedElements = inputElements.reshape(tensorType);
    Value constVal = create.onnx.constant(unsqueezedElements);
    return constVal;
  } else {
    return rewriter
        .create<ONNXUnsqueezeOp>(loc, tensorType, create.onnx.toTensor(input),
            create.onnx.constantInt64({axis}))
        .getResult();
  }
}

/// Emit an ONNXSplitOp. If the input is constant, do const propagation, and
/// return constants.
/// Only support evenly splitting.
std::vector<Value> foldOrEmitONNXSplitOpStableHlo(
    ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<Type> resultTypes, Value input, int64_t axis) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  std::vector<Value> resVals;
  int outputNum = resultTypes.size();
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    auto inputShape = inputElements.getType().getShape();
    assert(outputNum == 0 || inputShape[axis] % outputNum == 0);
    int64_t sizeOfEachSplit = outputNum != 0 ? inputShape[axis] / outputNum : 0;
    SmallVector<int64_t, 4> sizes(outputNum, sizeOfEachSplit);

    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    std::vector<ElementsAttr> splits =
        elementsBuilder.split(inputElements, axis, sizes);
    for (ElementsAttr splitElements : splits) {
      // Avoid DisposableElementsAttr during conversion.
      DenseElementsAttr denseSplitElements =
          elementsBuilder.toDenseElementsAttr(splitElements);
      Value constVal = create.onnx.constant(denseSplitElements);
      resVals.emplace_back(constVal);
    }
  } else {
    SmallVector<Type, 4> convertedTypes;
    SmallVector<int64_t> splitSizesI64;
    for (auto t : resultTypes) {
      convertedTypes.emplace_back(create.onnx.toTensor(t));
      splitSizesI64.emplace_back(t.cast<ShapedType>().getShape()[axis]);
    }
    Value splitSizes = create.onnx.constantInt64(splitSizesI64);
    ONNXSplitOp split = rewriter.create<ONNXSplitOp>(loc, convertedTypes,
        create.onnx.toTensor(input), splitSizes,
        /*axis=*/axis, nullptr);
    for (int i = 0; i < outputNum; ++i)
      resVals.emplace_back(split.getOutputs()[i]);
  }
  return resVals;
}

/// Emit an ONNXTransposeOp. If the input is constant, do const propagation,
/// and return a constant.
Value foldOrEmitONNXTransposeOpStableHlo(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, ArrayAttr permAttr) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    SmallVector<uint64_t, 4> perm;
    for (auto permVal : permAttr.getValue())
      perm.emplace_back(permVal.cast<IntegerAttr>().getInt());

    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    ElementsAttr transposedElements =
        elementsBuilder.transpose(inputElements, perm);
    // Avoid DisposableElementsAttr during conversion.
    DenseElementsAttr denseTransposedElements =
        elementsBuilder.toDenseElementsAttr(transposedElements);
    Value constVal = create.onnx.constant(denseTransposedElements);
    return constVal;
  } else {
    return rewriter
        .create<ONNXTransposeOp>(loc, create.onnx.toTensor(resultType),
            create.onnx.toTensor(input), permAttr)
        .getResult();
  }
}

} // namespace onnx_mlir
