//===--------------- LSTM.cpp - Lowering LSTM Op --------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LSTM Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

static const std::string FORWARD = "forward";
static const std::string REVERSE = "reverse";
static const std::string BIDIRECTIONAL = "bidirectional";

template <typename RNNOp>
struct ONNXRNNOpLowering : public ConversionPattern {
  ONNXRNNOpLowering(MLIRContext *ctx)
      : ConversionPattern(RNNOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Match
    RNNOp rnnOp = llvm::dyn_cast<RNNOp>(op);

    Value X = rnnOp.X();
    Value W = rnnOp.W();
    Value R = rnnOp.R();

    Type yTy = op->getResultTypes()[0];
    Type yhTy = op->getResultTypes()[1];
    Type ycTy = op->getResultTypes()[2];

    bool needAllStates = yTy.isa<NoneType>() ? false : true;
    bool needLastState = yhTy.isa<NoneType>() ? false : true;
    bool needCellState = ycTy.isa<NoneType>() ? false : true;

    // Delete this op if there is no output.
    if (!needAllStates && !needLastState && !needCellState) {
      rewriter.eraseOp(op);
      return success();
    }

    // Rewrite
    auto xShape = X.getType().cast<ShapedType>().getShape();
    auto wShape = W.getType().cast<ShapedType>().getShape();
    auto rShape = R.getType().cast<ShapedType>().getShape();

    auto sequenceLengthDim = xShape[0];
    auto batchSizeDim = xShape[1];
    auto inputSizeDim = xShape[2];
    auto numDirectionDim = rShape[0];
    auto hiddenSizeDim = rShape[2];

    auto direction = rnnOp.direction();

    // Insert an allocation and deallocation for the results of this operation.
    MemRefType yMemRefType, yhMemRefType, ycMemRefType;
    if (needAllStates)
      yMemRefType = convertToMemRefType(yTy);
    if (needLastState)
      yhMemRefType = convertToMemRefType(yhTy);
    if (needCellState)
      ycMemRefType = convertToMemRefType(ycTy);

    Value allocAllStates, allocLastState, allocCellState;
    if (needAllStates) {
      if (hasAllConstantDimensions(yMemRefType))
        allocAllStates = insertAllocAndDealloc(
            yMemRefType, loc, rewriter, checkInsertDealloc(op, 0));
      else
        // TODO: add code.
        emitError(loc, "Unsupported dynamic dimensions.");
    }

    if (needLastState) {
      if (hasAllConstantDimensions(yhMemRefType))
        allocLastState = insertAllocAndDealloc(
            yhMemRefType, loc, rewriter, checkInsertDealloc(op, 1));
      else
        // TODO: add code.
        emitError(loc, "Unsupported dynamic dimensions.");
    }

    if (needCellState) {
      if (hasAllConstantDimensions(ycMemRefType))
        allocCellState = insertAllocAndDealloc(
            ycMemRefType, loc, rewriter, checkInsertDealloc(op, 2));
      else
        // TODO: add code.
        emitError(loc, "Unsupported dynamic dimensions.");
    }

    rewriter.replaceOp(op, {allocAllStates, allocLastState, allocCellState});

    return success();
  }
};

void populateLoweringONNXLSTMOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXLSTMOp>>(ctx);
}
