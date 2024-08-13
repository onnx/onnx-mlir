/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ Expand.cpp - Expand Op ---------------------===//
//
// Copyright (c) 2024 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX ExpandOp to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrHelper.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include <cstdint>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXExpandLoweringToTOSA : public OpConversionPattern<ONNXExpandOp> {

public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXExpandOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXExpandOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto shape = adaptor.getShape();
    DenseIntElementsAttr denseAttr;
    if (!matchPattern(shape, m_Constant(&denseAttr))) {
      return rewriter.notifyMatchFailure(
          op, "onnx.expand can only be lowered with constant expanded shape");
    }

    // Convert denseAttr to DenseI64ArrayAttr. This handles both splat and
    // non-splat scenarios.
    ArrayBuffer<WideNum> shapeWideNums = getElementsWideNums(denseAttr);
    ArrayRef<int64_t> shapeArray =
        castArrayRef<int64_t, WideNum>(shapeWideNums.get());

    auto inputType =
        llvm::dyn_cast_or_null<RankedTensorType>(adaptor.getInput().getType());
    auto outputType =
        llvm::dyn_cast_or_null<RankedTensorType>(op.getResult().getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "Unranked and dynamic types are not supported");
    }
    size_t inputRank = onnx_mlir::getRank(inputType);

    // If inputRank is inferior to shapeRank we need to introduce a
    // reshape before the tile
    auto newInput = adaptor.getInput();
    if (inputRank != shapeArray.size()) {
      llvm::SmallVector<int64_t> newShape =
          getNewShape(inputType.getShape(), outputType.getShape());
      // If the newShape size doesn't match the output shape size, it means we
      // didn't find a proper reshape to match the input to.
      if (newShape.size() != outputType.getShape().size()) {
        return rewriter.notifyMatchFailure(
            op, "Could not find a shape that satisfies the expand constraints");
      }
      TosaBuilder tosaBuilder(rewriter, op->getLoc());
      newInput = tosaBuilder.reshape(adaptor.getInput(), newShape);
    }

    auto denseShape =
        getMultiplies(op, cast<RankedTensorType>(newInput.getType()).getShape(),
            outputType.getShape());
    auto resultElementType = cast<RankedTensorType>(inputType).getElementType();
    auto newResultElementType =
        getTypeConverter()->convertType(resultElementType);

    if (!isSupportedElementType(newResultElementType)) {
      return rewriter.notifyMatchFailure(
          op, "input/output type is invalid for tosa.tile");
    }
    Type newTileOutputType = RankedTensorType::get(
        llvm::SmallVector<int64_t>(
            outputType.getShape().size(), ShapedType::kDynamic),
        newResultElementType);
    onnx_mlir::tosa::CreateReplaceOpAndInfer<mlir::tosa::TileOp>(
        rewriter, op, newTileOutputType, newInput, denseShape);
    return success();
  }

private:
  // Supported element types for tosa.tile
  static bool isSupportedElementType(Type type) {
    if (auto intTy = dyn_cast_or_null<IntegerType>(type)) {
      // Supported integer bit widths
      std::set<unsigned> intWidth({8, 16, 32});
      return isTOSABool(type) ||
             (intTy.isSignless() &&
                 (intWidth.find(intTy.getWidth()) != intWidth.end()));
    }
    return type.isBF16() || type.isF16() || type.isF32();
  }

  static llvm::SmallVector<int64_t> getNewShape(
      const llvm::ArrayRef<int64_t> &inputShape,
      const llvm::ArrayRef<int64_t> &outputShape) {
    llvm::SmallVector<int64_t> result;
    size_t inputIdx = 0;
    for (auto outputDimension : outputShape) {
      // - If the inputIdx goes beyond the input shape, it means we are
      // extending the shape:
      //    Ex: 1x3x4 -> 1x3x4x1
      // - If the input dim is < output dim and do not divide it,
      // it's a dimension being added:
      //    Ex: 3x1 -> 2x1x6 (first dim is a new dim and not a tiled original
      //    one)
      // - If the output dim is < input dim,
      // it's also a dim being added:
      //    Ex: 2x3x4 -> 1x2x3x4
      if (inputIdx >= inputShape.size() ||
          (inputShape[inputIdx] < outputDimension &&
              outputDimension % inputShape[inputIdx] != 0) ||
          outputDimension < inputShape[inputIdx]) {
        result.push_back(1);
      } else {
        result.push_back(inputShape[inputIdx]);
        inputIdx++;
      }
    }
    return result;
  }

  static DenseI64ArrayAttr getMultiplies(ONNXExpandOp &op,
      const llvm::ArrayRef<int64_t> &inputShape,
      const llvm::ArrayRef<int64_t> &outputShape) {
    llvm::SmallVector<int64_t> multipliesArray;
    for (size_t i = 0; i < outputShape.size(); ++i) {
      if (i >= inputShape.size() || outputShape[i] / inputShape[i] == 0) {
        multipliesArray.push_back(1);
      } else {
        multipliesArray.push_back(outputShape[i] / inputShape[i]);
      }
    }
    return DenseI64ArrayAttr::get(op.getContext(), multipliesArray);
  }
};

} // namespace

void populateLoweringONNXExpandOpToTOSAPattern(ConversionTarget & /*target*/,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXExpandLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
