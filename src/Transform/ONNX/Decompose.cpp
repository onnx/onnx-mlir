/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to decompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the decomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

// Create an DenseElementsAttr of ArrayAttr.
// This function is used to get Value Type of an EXISTING ArrayAttr for Scaler
// function.
DenseElementsAttr createDenseArrayAttr(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  mlir::Type elementType = rewriter.getF32Type();
  int nElements = origAttrs.getValue().size();
  SmallVector<float, 4> wrapper(nElements, 0);
  for (int i = 0; i < nElements; ++i) {
    wrapper[i] = origAttrs.getValue()[i].cast<FloatAttr>().getValueAsDouble();
  }
  return DenseElementsAttr::get(
      RankedTensorType::get(wrapper.size(), elementType),
      llvm::makeArrayRef(wrapper));
}

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXDecompose.inc"

namespace {

RankedTensorType getReductionOutputType(RankedTensorType operandTy,
    ArrayRef<int64_t> reductionAxes, bool keepdims = true) {
  int64_t rank = operandTy.getRank();

  SmallVector<int64_t, 4> dims;
  for (decltype(rank) i = 0; i < rank; ++i) {
    if (std::find(reductionAxes.begin(), reductionAxes.end(), i) !=
        reductionAxes.end()) {
      if (keepdims)
        dims.emplace_back(1); // reduction dimension
    } else {
      dims.emplace_back(operandTy.getShape()[i]);
    }
  }

  return RankedTensorType::get(dims, operandTy.getElementType());
}

// Layer Normalization does an element-by-element normalization on the elements
// in the given axis. Where X is the set of input elements, this looks something
// like:
//                 (X - mean(X)) * scale
//          y = --------------------------- + bias
//              sqrt(variance(X) + epsilon)
struct ONNXLayerNormalizationOpPattern : public ::mlir::RewritePattern {
  ONNXLayerNormalizationOpPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("onnx.LayerNormalization", 1, context) {}
  ::mlir::LogicalResult matchAndRewrite(
      ::mlir::Operation *op, ::mlir::PatternRewriter &rewriter) const override {
    auto layerOp = llvm::dyn_cast<ONNXLayerNormalizationOp>(op);
    auto loc = op->getLoc();

    auto outType = op->getResultTypes()[0].dyn_cast<TensorType>();
    auto reductionType = outType;
    if (op->getResultTypes()[0].isa<RankedTensorType>()) {
      auto rankedType = op->getResultTypes()[0].dyn_cast<RankedTensorType>();
      int64_t axis = layerOp.axis();
      if (axis < 0) {
        axis += rankedType.getRank();
      }
      reductionType = getReductionOutputType(rankedType, {axis});
    }

    auto axisAttr = IntegerAttr::get(
        IntegerType::get(op->getContext(), 64), layerOp.axis());
    auto reductionDim = ArrayAttr::get(op->getContext(), {axisAttr});

    auto meanOp = rewriter.create<ONNXReduceMeanOp>(
        loc, reductionType, layerOp.data(), reductionDim);
    auto diffOp = rewriter.create<ONNXSubOp>(loc, layerOp.data(), meanOp);

    auto diffSquaredOp = rewriter.create<ONNXMulOp>(loc, diffOp, diffOp);
    auto varianceOp = rewriter.create<ONNXReduceMeanOp>(
        loc, reductionType, diffSquaredOp, reductionDim);

    // Epsilon is constrained to be an f32 type while element type isn't
    auto epsilon = layerOp.epsilon();
    if (layerOp.epsilonAttr().getType() != outType.getElementType()) {
      const llvm::fltSemantics &semantics = outType.getElementType().isBF16()
                                                ? APFloat::BFloat()
                                                : APFloat::IEEEsingle();
      bool losesInfo;
      epsilon.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
      if (losesInfo) {
        op->emitWarning("Lost precision in converting epsilon to element type");
      }
    }
    auto epsilonType =
        RankedTensorType::get({1}, reductionType.getElementType());
    auto epsilonAttr = DenseElementsAttr::get(epsilonType, epsilon);
    auto epsilonOp = rewriter.create<ONNXConstantOp>(loc, epsilonType, nullptr,
        epsilonAttr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    Value denomOp = rewriter.create<ONNXAddOp>(loc, varianceOp, epsilonOp);
    denomOp = rewriter.create<ONNXSqrtOp>(loc, denomOp);

    Value normOp = rewriter.create<ONNXDivOp>(loc, diffOp, denomOp);
    normOp = rewriter.create<ONNXMulOp>(loc, outType, normOp, layerOp.weight());
    if (!layerOp.bias().getType().isa<NoneType>()) {
      normOp = rewriter.create<ONNXAddOp>(loc, outType, normOp, layerOp.bias());
    }

    op->getResult(0).replaceAllUsesWith(normOp);
    if (!layerOp.saved_mean().getType().isa<NoneType>()) {
      op->getResult(1).replaceAllUsesWith(meanOp);
    }

    if (!layerOp.saved_inv_std_var().getType().isa<NoneType>()) {
      auto one = rewriter.getI64IntegerAttr(1);
      auto oneOp = rewriter.create<ONNXConstantOp>(loc, varianceOp.getType(),
          nullptr, one, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
      auto invVarOp = rewriter.create<ONNXDivOp>(loc, oneOp, varianceOp);
      op->getResult(2).replaceAllUsesWith(invVarOp);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct DecomposeONNXToONNXPass
    : public PassWrapper<DecomposeONNXToONNXPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void DecomposeONNXToONNXPass::runOnFunction() {
  auto function = getFunction();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect>();

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  target.addIllegalOp<ONNXReduceL1Op>();
  target.addIllegalOp<ONNXReduceL2Op>();
  target.addIllegalOp<ONNXReduceLogSumOp>();
  target.addIllegalOp<ONNXReduceLogSumExpOp>();
  target.addIllegalOp<ONNXReduceSumSquareOp>();
  target.addIllegalOp<ONNXScalerOp>();
  target.addIllegalOp<ONNXLogSoftmaxOp>();
  target.addIllegalOp<ONNXLayerNormalizationOp>();

  OwningRewritePatternList patterns(context);
  populateWithGenerated(patterns);
  patterns.insert<ONNXLayerNormalizationOpPattern>(context);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
} // end anonymous namespace

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createDecomposeONNXToONNXPass() {
  return std::make_unique<DecomposeONNXToONNXPass>();
}
