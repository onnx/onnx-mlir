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

// Check a Value's type is none or not.
bool isNoneType(Value val);

// Get a dimension of the tensor's shape.
int64_t dimAt(Value val, int index);

// Apply an activation function on a given scalar operand.
Value applyActivation(ConversionPatternRewriter &rewriter, Location loc,
    RNNActivation activation, Value scalarOperand);

// Override the following methods when lowering an RNN operation:
// - hasAllNoneOutput
// - getActivationPack
// - allocAndInitializeStates
// - calculateState
// - stateToOutput

// Check whether all outputs have NoneType or not.
template <typename RNNOp>
bool hasAllNoneOutput(RNNOp *op);

// Obtain activations functions for a specific operation.
template <typename RNNOp, typename A>
std::tuple<A, A> getActivationPack(RNNOp *op);

// Allocate memory for RNN states and initialize them.
template <typename RNNOp, typename S>
S allocAndInitializeStates(ConversionPatternRewriter &rewriter, Location loc,
    RNNOp *op, OperandAdaptor<RNNOp> operandAdaptor);

// Calculate new states from the current input and states.
template <typename RNNOp, typename S, typename A>
void calculateState(ConversionPatternRewriter &rewriter, Location loc,
    OperandAdaptor<RNNOp> operandAdaptor, S state, A activationSet,
    Value directionIV, Value sequenceIV);

// Write states to the RNN's outputs.
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

    int64_t sequenceDimSize = dimAt(rnnOp.X(), 0);
    auto direction = rnnOp.direction();

    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      BuildKrnlLoop sequenceLoops(rewriter, loc, 1);
      sequenceLoops.createDefineOp();
      sequenceLoops.pushBounds(0, sequenceDimSize);
      sequenceLoops.createIterateOp();

      auto ipSequenceLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(sequenceLoops.getIterateBlock());
      {
        Value directionIV =
            emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
        Value sequenceIV = sequenceLoops.getInductionVar(0);
        // Emit calculation for one RNN step.
        calculateState<RNNOp, S, A>(rewriter, loc, operandAdaptor, state,
            activationForward, directionIV, sequenceIV);
      }
      rewriter.restoreInsertionPoint(ipSequenceLoops);
    }

    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      BuildKrnlLoop sequenceLoops(rewriter, loc, 1);
      sequenceLoops.createDefineOp();
      sequenceLoops.pushBounds(0, sequenceDimSize);
      sequenceLoops.createIterateOp();

      auto ipSequenceLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(sequenceLoops.getIterateBlock());
      {
        AffineMap reverseIVMap = AffineMap::get(1, 1,
            rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0) - 1);

        Value directionIV = emitConstantOp(rewriter, loc,
            rewriter.getIndexType(), (direction == REVERSE) ? 0 : 1);
        Value reverseSequenceIV =
            rewriter.create<AffineApplyOp>(loc, reverseIVMap,
                std::vector<Value>{sequenceLoops.getInductionVar(0),
                    emitConstantOp(rewriter, loc, rewriter.getIndexType(),
                        sequenceDimSize)});
        // Emit calculation for one RNN step.
        calculateState<RNNOp, S, A>(rewriter, loc, operandAdaptor, state,
            activationReverse, directionIV, reverseSequenceIV);
      }
      rewriter.restoreInsertionPoint(ipSequenceLoops);
    }

    std::vector<Value> outputs;
    stateToOutput<RNNOp, S>(&rnnOp, state, outputs);
    rewriter.replaceOp(op, outputs);
    return success();
  }
};
