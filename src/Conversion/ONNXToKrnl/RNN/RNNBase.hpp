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

static const StringRef FORWARD = "forward";
static const StringRef REVERSE = "reverse";
static const StringRef BIDIRECTIONAL = "bidirectional";

struct RNNActivation {
  StringRef name;
  Optional<FloatAttr> alpha;
  Optional<FloatAttr> beta;
};

Value applyActivation(ConversionPatternRewriter &rewriter, Location loc,
    RNNActivation activation, Value input, Type elementType);

template <typename RNNOp, typename I, typename O>
std::tuple<I, O> getInputOutputPack(Operation *op, ArrayRef<Value> operands);

template <typename RNNOp, typename A>
std::tuple<A, A> getActivationPack(Operation *op);

template <typename O>
bool hasNoOutput(O outputPack);

template <typename I, typename O, typename S>
S allocAndInitializeStates(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, I inputPack, O outputPack);

template <typename RNNOp, typename I, typename S, typename A>
void calculateState(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Value numDirectionIV, Value sequenceLengthIV, I inputPack,
    S state, A activationSet);

template <typename RNNOp, typename S, typename O>
void stateToOutput(S state, O outputPack, std::vector<Value> &outputs);

template <typename RNNOp, typename I, typename O, typename S, typename A>
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
    A activationForward, activationReverse;
    std::tie(activationForward, activationReverse) =
        getActivationPack<RNNOp, A>(op);

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
            sequenceLengthIV, inputPack, state, activationForward);
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
            reverseSequenceLengthIV, inputPack, state, activationReverse);
      }
      rewriter.restoreInsertionPoint(ipSequenceLoops);
    }

    std::vector<Value> outputs;
    stateToOutput<RNNOp, S, O>(state, outputPack, outputs);
    rewriter.replaceOp(op, llvm::makeArrayRef(outputs));
    return success();
  }
};
