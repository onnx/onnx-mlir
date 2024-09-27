/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- RNNBase.hpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines base functions for lowering the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_RNN_BASE_KRNL_H
#define ONNX_MLIR_RNN_BASE_KRNL_H

#include "mlir/IR/AffineExpr.h"

#include "src/Conversion/ONNXConversionCommon/RNN/RNNBase.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

static constexpr int BUFFER_ALIGN = 128;

namespace onnx_mlir {

/// Insert Allocate and Deallocate for the all hidden output.
mlir::Value allocAllHidden(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, const mlir::TypeConverter *typeConverter, mlir::Value X,
    mlir::Value W, mlir::Value R, mlir::Value output);

/// Insert Allocate and Deallocate for the hidden or cell output.
mlir::Value allocHiddenOrCell(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, const mlir::TypeConverter *typeConverter, mlir::Value X,
    mlir::Value W, mlir::Value R, mlir::Value output);

/// Initialize the hidden and cell states.
void initializeHiddenAndCell(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value ht, mlir::Value ct, mlir::Value initialH,
    mlir::Value initialC, mlir::Type elementType, bool onlyHidden = false);

/// Allocate the intermediate hidden or cell state.
mlir::Value allocIntermediateState(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value X, mlir::Value R);

/// Initialize the intermediate hidden and cell states.
void initializeIntermediateStates(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value forwardHt, mlir::Value reverseHt,
    mlir::Value forwardCt, mlir::Value reverseCt, mlir::Value initialH,
    mlir::Value initialC, mlir::Type elementType, llvm::StringRef direction,
    bool onlyHidden);

/// Store a state into the output of the RNN op.
/// The input state is 2D and the output state is 3D with '1' or '2' is
/// pretended, depending on 'direction'.
void stateToOutputForHiddenOrCell(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value forwardVal, mlir::Value reverseVal,
    llvm::StringRef direction, mlir::Value output);

/// Get a slice of X at a specific timestep.
mlir::Value emitXSliceAt(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value X, mlir::Value timestep);

// Change the nextHt and Ht value if sequenceLens is defined.
// When a sample reachs the limit of its sequence len, nextHt will be padded
// with 0 (or initialH), and Ht will keep the last value at the sequence end
// so that the final value Ht is the last value at their sequence len.
mlir::Value handleSequenceLens(const KrnlBuilder &createKrnl,
    const MathBuilder &createMath, mlir::Value sequenceLens,
    mlir::Value initialH, mlir::Value nextHt, mlir::Value sequenceIV,
    mlir::Value directionIV, mlir::Value bs, mlir::Value hs, mlir::Value Ht);

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
    mlir::Location loc, const mlir::TypeConverter *typeConverter, RNNOp *op,
    typename RNNOp::Adaptor operandAdaptor);

// Calculate new states from the current input and states.
template <typename S, typename A, typename W, typename B>
void calculateState(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value Xt, S state, A activationSet, W weight,
    B bias, mlir::Value sequenceIV, mlir::Value directionIV,
    mlir::Value sequenceLens, mlir::Value initialH, bool isForward);

// Write states to the RNN's outputs.
template <typename RNNOp, typename S>
void stateToOutput(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, RNNOp *op, S state, std::vector<mlir::Value> &outputs);

// A common template for lowering an RNN operation.
template <typename RNNOp, typename S, typename A, typename W, typename B>
struct ONNXRNNOpLowering : public mlir::OpConversionPattern<RNNOp> {
  using OpAdaptor = typename RNNOp::Adaptor;

  ONNXRNNOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<RNNOp>(typeConverter, ctx) {}

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
        rewriter, loc, this->typeConverter, &rnnOp, adaptor);

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

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      IndexExprScope childScope(create.krnl);
      IndexExpr lb = LitIE(0);
      IndexExpr ub;
      if (!mlir::ShapedType::isDynamic(sequenceDimSize))
        ub = LitIE(sequenceDimSize);
      else
        ub = create.krnlIE.getShapeAsDim(X, 0);
      create.krnl.forLoopIE(lb, ub, /*step*/ 1, /*par*/ false,
          [&](const KrnlBuilder &createKrnl, mlir::ValueRange loopInd) {
            MathBuilder createMath(createKrnl);
            mlir::Value directionIV =
                createMath.constant(rewriter.getIndexType(), 0);
            mlir::Value sequenceIV = loopInd[0];
            // Get a slice of X at the current timestep.
            mlir::Value Xt = emitXSliceAt(rewriter, loc, X, sequenceIV);
            // Emit calculation for one RNN step.
            calculateState<S, A, W, B>(rewriter, loc, Xt, state,
                activationForward, weightForward, biasForward, sequenceIV,
                directionIV, sequenceLens, initialH,
                /*isForward=*/true);
          });
    }

    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      IndexExprScope childScope(create.krnl);
      IndexExpr lb = LitIE(0);
      IndexExpr ub;
      if (!mlir::ShapedType::isDynamic(sequenceDimSize))
        ub = LitIE(sequenceDimSize);
      else
        ub = create.krnlIE.getShapeAsDim(X, 0);
      create.krnl.forLoopIE(lb, ub, /*step*/ 1, /*par*/ false,
          [&](const KrnlBuilder &ck, mlir::ValueRange loopInd) {
            MultiDialectBuilder<MemRefBuilder, MathBuilder> create(ck);

            mlir::AffineMap reverseIVMap = mlir::AffineMap::get(1, 1,
                rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0) -
                    1);

            mlir::Value directionIV = create.math.constant(
                rewriter.getIndexType(), (direction == REVERSE) ? 0 : 1);
            mlir::Value sequenceSize =
                (!mlir::ShapedType::isDynamic(sequenceDimSize))
                    ? create.math.constant(
                          rewriter.getIndexType(), sequenceDimSize)
                    : create.mem.dim(X, 0);

            mlir::Value reverseSequenceIV =
                rewriter.create<mlir::affine::AffineApplyOp>(loc, reverseIVMap,
                    std::vector<mlir::Value>{loopInd[0], sequenceSize});
            // Get a slice of X at the current timestep.
            mlir::Value Xt = emitXSliceAt(rewriter, loc, X, reverseSequenceIV);
            // Emit calculation for one RNN step.
            calculateState<S, A, W, B>(rewriter, loc, Xt, state,
                activationReverse, weightReverse, biasReverse,
                reverseSequenceIV, directionIV, sequenceLens, initialH,
                /*isForward=*/false);
          });
    }

    std::vector<mlir::Value> outputs;
    stateToOutput<RNNOp, S>(rewriter, loc, &rnnOp, state, outputs);
    rewriter.replaceOp(op, outputs);
    return mlir::success();
  }
};

} // namespace onnx_mlir
#endif
