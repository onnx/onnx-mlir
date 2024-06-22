/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Pad.cpp - Lowering Pad Op ------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers ONNX Pad Operators to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXPadOpLoweringToStablehlo : public ConversionPattern {
  ONNXPadOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXPadOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    ONNXPadOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Value data = operandAdaptor.getData();
    Value constantValue = operandAdaptor.getConstantValue();
    Value pads = operandAdaptor.getPads();
    StringRef padMode = operandAdaptor.getMode();

    if (!padMode.equals_insensitive("constant"))
      return failure();
    assert(isRankedShapedType(data.getType()) && "Expected Ranked ShapedType");
    ShapedType inputType = mlir::cast<ShapedType>(data.getType());
    Type elemType = inputType.getElementType();
    int64_t rank = inputType.getRank();

    Type outputType = *op->result_type_begin();
    if (!constantValue || isNoneValue(constantValue)) {
      // Pad with zeros by default
      constantValue = rewriter.create<stablehlo::ConstantOp>(
          loc, DenseElementsAttr::get(mlir::RankedTensorType::get({}, elemType),
                   rewriter.getZeroAttr(elemType)));
    } else {
      // constantValue might be 1D tensor, reshape it to scalar
      ShapedType constantType = mlir::cast<ShapedType>(constantValue.getType());
      if (constantType.getRank() != 0)
        constantValue = rewriter.create<stablehlo::ReshapeOp>(
            loc, RankedTensorType::get({}, elemType), constantValue);
    }
    SmallVector<int64_t> edgePaddingLowVec(rank, 0);
    SmallVector<int64_t> edgePaddingHighVec(rank, 0);
    SmallVector<int64_t> interiorPaddingVec(rank, 0);
    if (auto valueAttribute = getElementAttributeFromConstValue(pads)) {
      // If `pads` are constants, read them."
      int64_t idx = 0;
      for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
        int64_t padValue = value.getInt();
        if (padValue < 0)
          return failure();
        if (idx < rank)
          edgePaddingLowVec[idx] = padValue;
        else
          edgePaddingHighVec[idx - rank] = padValue;
        idx++;
      }
    } else {
      assert(false && "Pads must be known at compile time");
    }

    mlir::DenseI64ArrayAttr edgePaddingLow =
        rewriter.getDenseI64ArrayAttr(edgePaddingLowVec);
    mlir::DenseI64ArrayAttr edgePaddingHigh =
        rewriter.getDenseI64ArrayAttr(edgePaddingHighVec);
    mlir::DenseI64ArrayAttr interiorPadding =
        rewriter.getDenseI64ArrayAttr(interiorPaddingVec);
    Value padResult = rewriter.create<stablehlo::PadOp>(loc, outputType, data,
        constantValue, edgePaddingLow, edgePaddingHigh, interiorPadding);

    rewriter.replaceOp(op, padResult);
    return success();
  }
};

} // namespace

void populateLoweringONNXPadOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPadOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
