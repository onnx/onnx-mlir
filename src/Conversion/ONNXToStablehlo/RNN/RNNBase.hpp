/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- RNNBase.hpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2023-2024
//
// =============================================================================
//
// This file defines base functions for lowering the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_RNN_BASE_STABLEHLO_H
#define ONNX_MLIR_RNN_BASE_STABLEHLO_H

#include "src/Conversion/ONNXConversionCommon/RNN/RNNBase.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"

namespace onnx_mlir {

namespace stablehlo {

/// Allocate the all hidden output.
mlir::Value allocAllHidden(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value X, mlir::Value R);

/// Allocate the hidden or cell output.
mlir::Value allocHiddenOrCell(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value X, mlir::Value W, mlir::Value R);

/// Allocate the intermediate hidden or cell state.
mlir::Value allocIntermediateState(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value X, mlir::Value R);

/// Initialize the intermediate hidden and cell states.
void initializeIntermediateStates(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value &forwardHt, mlir::Value &reverseHt,
    mlir::Value &forwardCt, mlir::Value &reverseCt, mlir::Value initialH,
    mlir::Value initialC, mlir::Type elementType, llvm::StringRef direction,
    bool onlyHidden);

/// Store a state into the output of the RNN op.
/// The input state is 2D and the output state is 3D with '1' or '2' is
/// pretended, depending on 'direction'.
void stateToOutputForHiddenOrCell(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value forwardVal, mlir::Value reverseVal,
    llvm::StringRef direction, mlir::Value &output);

/// Get a slice of X at a specific timestep.
mlir::Value emitXSliceAt(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value X, mlir::Value timestep);

// Override the following methods when lowering an RNN operation:
// - hasAllNoneOutput
// - getActivationPack
// - getWeightPack
// - getBiasPack
// - allocAndInitializeStates
// - calculateState
// - stateToOutput

/// Obtain weight tensors in 2D for each gate.
/// In ONNX, weights for gates and directions are combined in a single tensor.
/// This function splits them into 2D tensors.
template <typename RNNOp, typename W>
std::tuple<W, W> getWeightPack(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, RNNOp *op);

/// Obtain biases in 1D for each gate.
/// In ONNX, biases for gates and directions are combined in a single tensor.
/// This function splits them into 1D tensors.
template <typename RNNOp, typename B>
std::tuple<B, B> getBiasPack(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, RNNOp *op);

// Allocate memory for RNN states and initialize them.
template <typename RNNOp, typename S>
S allocAndInitializeStates(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, RNNOp *op, typename RNNOp::Adaptor operandAdaptor,
    bool enableUnroll);

// Calculate new states from the current input and states.
template <typename S, typename A, typename W, typename B>
void calculateState(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value Xt, S &state, A activationSet, W weight,
    B bias, mlir::Value sequenceIV, mlir::Value directionIV,
    mlir::Value sequenceLens, mlir::Value initialH, bool enableUnroll,
    bool isForward);

// Write states to the RNN's outputs.
template <typename RNNOp, typename S>
void stateToOutput(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, RNNOp *op, S state, std::vector<mlir::Value> &outputs,
    bool enableUnroll);

// Calculate all states using unroll
template <typename RNNOp, typename S, typename A, typename W, typename B>
void calculateStateWithUnroll(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, llvm::StringRef direction, int64_t sequenceDimSize,
    Value X, S &state, A activationForward, A activationReverse,
    W weightForward, W weightReverse, B biasForward, B biasReverse,
    Value sequenceLens, Value initialH);

// Calculate all states using loop
template <typename RNNOp, typename S, typename A, typename W, typename B>
void calculateStateWithLoop(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, llvm::StringRef direction, int64_t sequenceDimSize,
    Value X, S &state, A activationForward, A activationReverse,
    W weightForward, W weightReverse, B biasForward, B biasReverse,
    Value sequenceLens, Value initialH);

// A common template for lowering an RNN operation.
template <typename RNNOp, typename S, typename A, typename W, typename B>
struct ONNXRNNOpLowering : public mlir::OpConversionPattern<RNNOp> {
  using OpAdaptor = typename RNNOp::Adaptor;
  bool enableUnroll;

  ONNXRNNOpLowering(mlir::MLIRContext *ctx, bool enableUnroll)
      : mlir::OpConversionPattern<RNNOp>(ctx) {
    this->enableUnroll = enableUnroll;
  }

  mlir::LogicalResult matchAndRewrite(RNNOp rnnOp, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Operation *op = rnnOp.getOperation();
    mlir::Location loc = ONNXLoc<RNNOp>(op);
    mlir::Value X = adaptor.getX();
    mlir::Value sequenceLens = adaptor.getSequenceLens();
    mlir::Value initialH = adaptor.getInitialH();

    if (hasAllNoneOutput<RNNOp>(&rnnOp)) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Initialize output states.
    S state = allocAndInitializeStates<RNNOp, S>(
        rewriter, loc, &rnnOp, adaptor, this->enableUnroll);

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

    int64_t sequenceDimSize = dimAt(rnnOp.getX(), 0);
    auto direction = rnnOp.getDirection();

    if (this->enableUnroll)
      calculateStateWithUnroll<RNNOp, S, A, W, B>(rewriter, loc, direction,
          sequenceDimSize, X, state, activationForward, activationReverse,
          weightForward, weightReverse, biasForward, biasReverse, sequenceLens,
          initialH);
    else
      calculateStateWithLoop<RNNOp, S, A, W, B>(rewriter, loc, direction,
          sequenceDimSize, X, state, activationForward, activationReverse,
          weightForward, weightReverse, biasForward, biasReverse, sequenceLens,
          initialH);
    std::vector<mlir::Value> outputs;
    stateToOutput<RNNOp, S>(
        rewriter, loc, &rnnOp, state, outputs, this->enableUnroll);
    rewriter.replaceOp(op, outputs);
    return mlir::success();
  }
};

} // namespace stablehlo

} // namespace onnx_mlir
#endif
