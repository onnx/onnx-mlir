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
    auto padOp = mlir::dyn_cast<ONNXPadOp>(op);
    ONNXPadOpAdaptor operandAdaptor(operands, padOp);
    Value data = operandAdaptor.getData();
    Value constantValue = operandAdaptor.getConstantValue();
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
      constantValue = stablehlo::ConstantOp::create(rewriter, loc,
          DenseElementsAttr::get(mlir::RankedTensorType::get({}, elemType),
              rewriter.getZeroAttr(elemType)));
    } else {
      // constantValue might be 1D tensor, reshape it to scalar
      ShapedType constantType = mlir::cast<ShapedType>(constantValue.getType());
      if (constantType.getRank() != 0)
        constantValue = stablehlo::ReshapeOp::create(
            rewriter, loc, RankedTensorType::get({}, elemType), constantValue);
    }
    SmallVector<int64_t> edgePaddingLowVec(rank, 0);
    SmallVector<int64_t> edgePaddingHighVec(rank, 0);
    SmallVector<int64_t> interiorPaddingVec(rank, 0);
    // Get pads per axis of `data` via the shape helper, which accounts for
    // `axes`; it fails when some axes are not known.
    ONNXPadOpShapeHelper shapeHelper(op, {});
    if (failed(shapeHelper.computeShape()))
      return rewriter.notifyMatchFailure(op, "Pad axes are not all known");
    for (int64_t i = 0; i < rank; ++i) {
      IndexExpr padBegin = shapeHelper.pads[i];
      IndexExpr padEnd = shapeHelper.pads[i + rank];
      assert(padBegin.isLiteral() && padEnd.isLiteral() &&
             "Pads must be known at compile time");
      if (padBegin.getLiteral() < 0 || padEnd.getLiteral() < 0)
        return failure();
      edgePaddingLowVec[i] = padBegin.getLiteral();
      edgePaddingHighVec[i] = padEnd.getLiteral();
    }

    mlir::DenseI64ArrayAttr edgePaddingLow =
        rewriter.getDenseI64ArrayAttr(edgePaddingLowVec);
    mlir::DenseI64ArrayAttr edgePaddingHigh =
        rewriter.getDenseI64ArrayAttr(edgePaddingHighVec);
    mlir::DenseI64ArrayAttr interiorPadding =
        rewriter.getDenseI64ArrayAttr(interiorPaddingVec);
    Value padResult = stablehlo::PadOp::create(rewriter, loc, outputType, data,
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
