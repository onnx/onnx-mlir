/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- RNNBase.hpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines base functions for lowerng the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

static constexpr int BUFFER_ALIGN = 128;
static constexpr StringRef FORWARD = "forward";
static constexpr StringRef REVERSE = "reverse";
static constexpr StringRef BIDIRECTIONAL = "bidirectional";

namespace onnx_mlir {

struct RNNActivation {
  StringRef name;
  Optional<FloatAttr> alpha;
  Optional<FloatAttr> beta;
};

/// Check a Value's type is none or not.
bool isNoneType(Value val);

/// Get a dimension of the tensor's shape.
int64_t dimAt(Value val, int index);

/// Insert Allocate and Deallocate for the all hidden output.
Value allocAllHidden(ConversionPatternRewriter &rewriter, Location loc, Value X,
    Value W, Value R, Value output, bool insertDealloc = false);

/// Insert Allocate and Deallocate for the hidden or cell output.
Value allocHiddenOrCell(ConversionPatternRewriter &rewriter, Location loc,
    Value X, Value W, Value R, Value output, bool insertDealloc = false);

/// Initialize the hidden and cell states.
void initializeHiddenAndCell(ConversionPatternRewriter &rewriter, Location loc,
    Value ht, Value ct, Value initialH, Value initialC, Type elementType,
    bool onlyHidden = false);

/// Allocate the intermediate hidden or cell state.
Value allocIntermediateState(
    ConversionPatternRewriter &rewriter, Location loc, Value X, Value R);

/// Initialize the intermediate hidden and cell states.
void initializeIntermediateStates(ConversionPatternRewriter &rewriter,
    Location loc, Value forwardHt, Value reverseHt, Value forwardCt,
    Value reverseCt, Value initialH, Value initialC, Type elementType,
    StringRef direction, bool onlyHidden);

/// Store a state into the output of the RNN op.
/// The input state is 2D and the output state is 3D with '1' or '2' is
/// pretended, depending on 'direction'.
void stateToOutputForHiddenOrCell(ConversionPatternRewriter &rewriter,
    Location loc, Value forwardVal, Value reverseVal, StringRef direction,
    Value output);

/// Apply an activation function on a given operand.
Value applyActivation(
    OpBuilder &rewriter, Location loc, RNNActivation activation, Value operand);

/// Get a slice of X at a specific timestep.
Value emitXSliceAt(
    ConversionPatternRewriter &rewriter, Location loc, Value X, Value timestep);

// Override the following methods when lowering an RNN operation:
// - hasAllNoneOutput
// - getActivationPack
// - getWeightPack
// - getBiasPack
// - allocAndInitializeStates
// - calculateState
// - stateToOutput

// Check whether all outputs have NoneType or not.
template <typename RNNOp>
bool hasAllNoneOutput(RNNOp *op);

// Obtain activations functions for a specific operation.
template <typename RNNOp, typename A>
std::tuple<A, A> getActivationPack(RNNOp *op);

/// Obtain weight tensors in 2D for each gate.
/// In ONNX, weights for gates and directions are combined in a single tensor.
/// This function splits them into 2D tensors.
template <typename RNNOp, typename W>
std::tuple<W, W> getWeightPack(
    ConversionPatternRewriter &rewriter, Location loc, RNNOp *op);

/// Obtain biases in 1D for each gate.
/// In ONNX, biases for gates and directions are combined in a single tensor.
/// This function splits them into 1D tensors.
template <typename RNNOp, typename B>
std::tuple<B, B> getBiasPack(
    ConversionPatternRewriter &rewriter, Location loc, RNNOp *op);

// Allocate memory for RNN states and initialize them.
template <typename RNNOp, typename S>
S allocAndInitializeStates(ConversionPatternRewriter &rewriter, Location loc,
    RNNOp *op, typename RNNOp::Adaptor operandAdaptor);

// Calculate new states from the current input and states.
template <typename S, typename A, typename W, typename B>
void calculateState(ConversionPatternRewriter &rewriter, Location loc, Value Xt,
    S state, A activationSet, W weight, B bias, Value sequenceIV,
    Value directionIV, bool isForward);

// Write states to the RNN's outputs.
template <typename RNNOp, typename S>
void stateToOutput(ConversionPatternRewriter &rewriter, Location loc, RNNOp *op,
    S state, std::vector<Value> &outputs);

// A common template for lowering an RNN operation.
template <typename RNNOp, typename S, typename A, typename W, typename B>
struct ONNXRNNOpLowering : public ConversionPattern {
  ONNXRNNOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, RNNOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    RNNOp rnnOp = llvm::dyn_cast<RNNOp>(op);
    typename RNNOp::Adaptor operandAdaptor(operands);
    Value X = operandAdaptor.X();

    if (hasAllNoneOutput<RNNOp>(&rnnOp)) {
      rewriter.eraseOp(op);
      return success();
    }

    // Initialize output states.
    S state = allocAndInitializeStates<RNNOp, S>(
        rewriter, loc, &rnnOp, operandAdaptor);

    // Activation functions.
    A activationForward, activationReverse;
    std::tie(activationForward, activationReverse) =
        getActivationPack<RNNOp, A>(&rnnOp);

    // Prepare weights.
    W weightForward, weightReverse;
    std::tie(weightForward, weightReverse) =
        getWeightPack<RNNOp, W>(rewriter, loc, &rnnOp);

    // Prepare biases.
    B biasForward, biasReverse;
    std::tie(biasForward, biasReverse) =
        getBiasPack<RNNOp, B>(rewriter, loc, &rnnOp);

    int64_t sequenceDimSize = dimAt(rnnOp.X(), 0);
    auto direction = rnnOp.direction();

    MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);

    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      krnl::BuildKrnlLoop sequenceLoops(rewriter, loc, 1);
      sequenceLoops.createDefineOp();
      if (sequenceDimSize != -1)
        sequenceLoops.pushBounds(0, sequenceDimSize);
      else
        sequenceLoops.pushBounds(0, create.mem.dim(X, 0));
      sequenceLoops.createIterateOp();

      auto ipSequenceLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(sequenceLoops.getIterateBlock());
      {
        Value directionIV =
            emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
        Value sequenceIV = sequenceLoops.getInductionVar(0);
        // Get a slice of X at the current timestep.
        Value Xt = emitXSliceAt(rewriter, loc, X, sequenceIV);
        // Emit calculation for one RNN step.
        calculateState<S, A, W, B>(rewriter, loc, Xt, state, activationForward,
            weightForward, biasForward, sequenceIV, directionIV,
            /*isForward=*/true);
      }
      rewriter.restoreInsertionPoint(ipSequenceLoops);
    }

    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      krnl::BuildKrnlLoop sequenceLoops(rewriter, loc, 1);
      sequenceLoops.createDefineOp();
      if (sequenceDimSize != -1)
        sequenceLoops.pushBounds(0, sequenceDimSize);
      else
        sequenceLoops.pushBounds(0, create.mem.dim(X, 0));
      sequenceLoops.createIterateOp();

      auto ipSequenceLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(sequenceLoops.getIterateBlock());
      {
        AffineMap reverseIVMap = AffineMap::get(1, 1,
            rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0) - 1);

        Value directionIV = emitConstantOp(rewriter, loc,
            rewriter.getIndexType(), (direction == REVERSE) ? 0 : 1);
        Value sequenceSize;
        if (sequenceDimSize != -1)
          sequenceSize = emitConstantOp(
              rewriter, loc, rewriter.getIndexType(), sequenceDimSize);
        else
          sequenceSize = create.mem.dim(X, 0);

        Value reverseSequenceIV = rewriter.create<AffineApplyOp>(loc,
            reverseIVMap,
            std::vector<Value>{sequenceLoops.getInductionVar(0), sequenceSize});
        // Get a slice of X at the current timestep.
        Value Xt = emitXSliceAt(rewriter, loc, X, reverseSequenceIV);
        // Emit calculation for one RNN step.
        calculateState<S, A, W, B>(rewriter, loc, Xt, state, activationReverse,
            weightReverse, biasReverse, reverseSequenceIV, directionIV,
            /*isForward=*/false);
      }
      rewriter.restoreInsertionPoint(ipSequenceLoops);
    }

    std::vector<Value> outputs;
    stateToOutput<RNNOp, S>(rewriter, loc, &rnnOp, state, outputs);
    rewriter.replaceOp(op, outputs);
    return success();
  }
};

} // namespace onnx_mlir
