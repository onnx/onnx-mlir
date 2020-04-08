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

Value applyActivation(ConversionPatternRewriter &rewriter, Location loc,
    std::string activation, Value input, Type elementType,
    Optional<float> alpha, Optional<float> beta);

Value activation_f(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, Type elementType);

Value activation_g(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, Type elementType);

Value activation_h(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value input, Type elementType);

template <typename RNNOp, typename I, typename O>
std::tuple<I, O> getInputOutputPack(Operation *op, ArrayRef<Value> operands);

template <typename O>
bool hasNoOutput(O outputPack);

template <typename I, typename O, typename S>
S allocAndInitializeStates(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, I inputPack, O outputPack);

template <typename RNNOp, typename I, typename S>
void calculateState(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value numDirectionIV, Value sequenceLengthIV, I inputPack,
    S state);

template <typename RNNOp, typename S, typename O>
void stateToOutput(S state, O outputPack, std::vector<Value> &outputs);

template <typename RNNOp, typename I, typename O, typename S>
struct ONNXRNNOpLowering : public ConversionPattern {
  ONNXRNNOpLowering(MLIRContext *ctx)
      : ConversionPattern(RNNOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    I inputPack;
    O outputPack;
    S state;

    std::tie(inputPack, outputPack) =
        getInputOutputPack<RNNOp, I, O>(op, operands);

    if (hasNoOutput<O>(outputPack)) {
      rewriter.eraseOp(op);
      return success();
    }

    state = allocAndInitializeStates<I, O, S>(
        rewriter, loc, op, inputPack, outputPack);

    Value X = inputPack.X;
    int sequenceLengthDim = X.getType().cast<ShapedType>().getShape()[0];
    auto direction = llvm::dyn_cast<RNNOp>(op).direction();

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
        Value sequenceLengthIV = sequenceLoops.getInductionVar(0);
        // Emit calculation for one RNN step.
        calculateState<RNNOp, I, S>(rewriter, loc, op, numDirectionIV,
            sequenceLengthIV, inputPack, state);
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
        Value reverseSequenceLengthIV = rewriter.create<SubIOp>(loc,
            emitConstantOp(
                rewriter, loc, rewriter.getIndexType(), sequenceLengthDim - 1),
            sequenceLoops.getInductionVar(0));
        // Emit calculation for one RNN step.
        calculateState<RNNOp, I, S>(rewriter, loc, op, numDirectionIV,
            reverseSequenceLengthIV, inputPack, state);
      }
      rewriter.restoreInsertionPoint(ipSequenceLoops);
    }

    std::vector<Value> outputs;
    stateToOutput<RNNOp, S, O>(state, outputPack, outputs);
    rewriter.replaceOp(op, llvm::makeArrayRef(outputs));
    return success();
  }
};
