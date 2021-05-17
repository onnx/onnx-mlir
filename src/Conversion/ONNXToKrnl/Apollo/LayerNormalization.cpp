/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ LayerNormalization.cpp - LayerNormalization Op ----------------===//
//
// Copyright 2021 Microsoft
//
// =============================================================================
//
// This file lowers ONNX LayerNormalization operator to a call.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace {

struct ONNXLayerNormalizationOpLowering : public ConversionPattern {

  static int nextCallId;

  ONNXLayerNormalizationOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXLayerNormalizationOp::getOperationName(), 1, ctx) {}

  // normalize the f32 epsilon attribute to match the target element type
  FloatAttr normalizeEpsilon(
      APFloat value, Type targetType, Location loc) const {
    assert(targetType.isBF16() || targetType.isF32());

    if (targetType.isBF16()) {
      const llvm::fltSemantics &semantics = APFloat::BFloat();
      bool losesInfo;
      value.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
      if (losesInfo) {
        emitWarning(loc, "Lost precision in converting epsilon to BF16 type");
      }
    }

    return FloatAttr::get(targetType, value);
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto ctx = getContext();
    auto layerNormOp = dyn_cast<ONNXLayerNormalizationOp>(op);
    if (!layerNormOp) {
      return failure();
    }

    auto dataType = layerNormOp.data().getType().dyn_cast<ShapedType>();
    auto axis = layerNormOp.axis();
    if (axis < 0) {
      assert(dataType.hasRank());
      axis += dataType.getRank();
    }

    if (axis != dataType.getRank() - 1 && axis != dataType.getRank() - 2) {
      return op->emitError("Only normalization on the last two axes supported");
    }

    const std::string funcName =
        llvm::formatv("tvp_LayerNormalization_{0}", nextCallId++);
    auto moduleOp = op->template getParentOfType<ModuleOp>();
    assert(!moduleOp.lookupSymbol<FuncOp>(funcName));

    auto resultTypes = op->getResultTypes();
    auto callOp = rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, funcName, resultTypes, operands);

    auto fnType = rewriter.getFunctionType(op->getOperandTypes(), resultTypes);
    // Insert the function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    rewriter.create<FuncOp>(moduleOp.getLoc(), funcName, fnType).setPrivate();

    // Call ops don't seem to retain custom attributes, so attach it to the
    // function itself
    Operation *funcOp = moduleOp.lookupSymbol<FuncOp>(funcName);
    funcOp->setAttr("tvp.layerNormalization", BoolAttr::get(ctx, true));

    funcOp->setAttr("axis", layerNormOp.axisAttr());

    // Epsilon is constrained to be an f32 type while element type isn't
    funcOp->setAttr("epsilon", normalizeEpsilon(layerNormOp.epsilon(),
                                   dataType.getElementType(), op->getLoc()));

    return success();
  }
};

int ONNXLayerNormalizationOpLowering::nextCallId = 0;
} // namespace

void populateLoweringONNXLayerNormalizationOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXLayerNormalizationOpLowering>(ctx);
}
