//===--------------- RNNBase.hpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines base functions for lowerng the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
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

// Apply an activation function on a given scalar operand.
Value applyActivation(ConversionPatternRewriter &rewriter, Location loc,
    RNNActivation activation, Value scalarOperand);

// Override the following methods when lowering an RNN operation:
// - hasAllNoneOutput
// - getActivationPack
// - allocAndInitializeStates
// - calculateState
// - stateToOutput

template <typename RNNOp>
bool hasAllNoneOutput(RNNOp *op);

template <typename RNNOp, typename A>
std::tuple<A, A> getActivationPack(RNNOp *op);

template <typename RNNOp, typename S>
S allocAndInitializeStates(ConversionPatternRewriter &rewriter, Location loc,
    RNNOp *op, OperandAdaptor<RNNOp> operandAdaptor);

template <typename RNNOp, typename S, typename A>
void calculateState(ConversionPatternRewriter &rewriter, Location loc,
    OperandAdaptor<RNNOp> operandAdaptor, S state, A activationSet,
    Value numDirectionIV, Value sequenceLengthIV);

template <typename RNNOp, typename S>
void stateToOutput(RNNOp *op, S state, std::vector<Value> &outputs);

// A common template for lowering an RNN operation.
template <typename RNNOp, typename S, typename A>
struct ONNXRNNOpLowering : public ConversionPattern {
  ONNXRNNOpLowering(MLIRContext *ctx)
      : ConversionPattern(RNNOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    RNNOp rnnOp = llvm::dyn_cast<RNNOp>(op);
    OperandAdaptor<RNNOp> operandAdaptor(operands);

    if (hasAllNoneOutput<RNNOp>(&rnnOp)) {
      rewriter.eraseOp(op);
      return success();
    }

    S state = allocAndInitializeStates<RNNOp, S>(
        rewriter, loc, &rnnOp, operandAdaptor);

    A activationForward, activationReverse;
    std::tie(activationForward, activationReverse) =
        getActivationPack<RNNOp, A>(&rnnOp);

    Value X = rnnOp.X();
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
        Value sequenceLengthIV = sequenceLoops.getInductionVar(0);
        // Emit calculation for one RNN step.
        calculateState<RNNOp, S, A>(rewriter, loc, operandAdaptor, state,
            activationForward, numDirectionIV, sequenceLengthIV);
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
        AffineMap reverseIVMap = AffineMap::get(1, 1,
            rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0) - 1);

        Value numDirectionIV = emitConstantOp(rewriter, loc,
            rewriter.getIndexType(), (direction == REVERSE) ? 0 : 1);
        Value reverseSequenceLengthIV =
            rewriter.create<AffineApplyOp>(loc, reverseIVMap,
                ValueRange(std::vector<Value>{sequenceLoops.getInductionVar(0),
                    emitConstantOp(rewriter, loc, rewriter.getIndexType(),
                        sequenceLengthDim)}));
        // Emit calculation for one RNN step.
        calculateState<RNNOp, S, A>(rewriter, loc, operandAdaptor, state,
            activationReverse, numDirectionIV, reverseSequenceLengthIV);
      }
      rewriter.restoreInsertionPoint(ipSequenceLoops);
    }

    std::vector<Value> outputs;
    stateToOutput<RNNOp, S>(&rnnOp, state, outputs);
    rewriter.replaceOp(op, outputs);
    return success();
  }
};
