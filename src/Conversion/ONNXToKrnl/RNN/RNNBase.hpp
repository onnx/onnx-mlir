//===--------------- RNNBase.hpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines base functions for lowerng the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

static const std::string FORWARD = "forward";
static const std::string REVERSE = "reverse";
static const std::string BIDIRECTIONAL = "bidirectional";

Value activation_f(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, Type elementType);

Value activation_g(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, Type elementType);

Value activation_h(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, Type elementType);

template <typename I>
I getInputPack(ArrayRef<Value> operands);

template <typename RNNOp>
bool hasNoOutput(Operation *op);

template <typename RNNOp, typename I, typename S>
S allocAndInitializeStates(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, I inputPack);

template <typename RNNOp, typename I, typename S>
void calculateState(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value numDirectionIV, Value sequenceIV, I inputPack,
    S state);

template <typename RNNOp, typename S>
void stateToOutput(Operation *op, S state, std::vector<Value> &outputs);

template <typename RNNOp, typename I, typename S>
struct ONNXRNNOpLowering : public ConversionPattern {
  ONNXRNNOpLowering(MLIRContext *ctx)
      : ConversionPattern(RNNOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    RNNOp rnnOp = llvm::dyn_cast<RNNOp>(op);

    if (hasNoOutput<RNNOp>(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    I inputPack = getInputPack<I>(operands);
    S state =
        allocAndInitializeStates<RNNOp, I, S>(rewriter, loc, op, inputPack);

    Value X = inputPack.X;
    int sequenceLengthDim = X.getType().cast<ShapedType>().getShape()[0];
    auto direction = rnnOp.direction();

    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      BuildKrnlLoop sequenceLoops(rewriter, loc, 1);
      sequenceLoops.createDefineAndOptimizeOp();
      sequenceLoops.pushBounds(0, sequenceLengthDim);
      sequenceLoops.createIterateOp();

      auto ipSequenceLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(sequenceLoops.getIterateBlock());
      {
        Value numDirectionIV =
            emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
        Value sequenceIV = sequenceLoops.getInductionVar(0);
        // Emit calculation for one RNN step.
        calculateState<RNNOp, I, S>(
            rewriter, loc, op, numDirectionIV, sequenceIV, inputPack, state);
      }
      rewriter.restoreInsertionPoint(ipSequenceLoops);
    }

    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      BuildKrnlLoop sequenceLoops(rewriter, loc, 1);
      sequenceLoops.createDefineAndOptimizeOp();
      sequenceLoops.pushBounds(0, sequenceLengthDim);
      sequenceLoops.createIterateOp();

      auto ipSequenceLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(sequenceLoops.getIterateBlock());
      {
        Value numDirectionIV = emitConstantOp(rewriter, loc,
            rewriter.getIndexType(), (direction == REVERSE) ? 0 : 1);
        Value reverseSequenceIV = rewriter.create<SubIOp>(loc,
            emitConstantOp(
                rewriter, loc, rewriter.getIndexType(), sequenceLengthDim - 1),
            sequenceLoops.getInductionVar(0));
        // Emit calculation for one RNN step.
        calculateState<RNNOp, I, S>(rewriter, loc, op, numDirectionIV,
            reverseSequenceIV, inputPack, state);
      }
      rewriter.restoreInsertionPoint(ipSequenceLoops);
    }

    std::vector<Value> outputs;
    stateToOutput<RNNOp, S>(op, state, outputs);
    rewriter.replaceOp(op, llvm::makeArrayRef(outputs));
    return success();
  }
};
