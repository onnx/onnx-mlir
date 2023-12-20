/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- RNNBase.cpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2023
//
// =============================================================================
//
// This file defines base functions for lowering the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStableHlo/RNN/RNNBase.hpp"
#include "src/Conversion/ONNXToStableHlo/ONNXToStableHloCommon.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lstm"

using namespace mlir;

namespace onnx_mlir {

namespace stablehlo {

// Get a dimension of the tensor's shape.
int64_t dimAt(Value val, int index) {
  return val.getType().cast<ShapedType>().getShape()[index];
}

/// Allocate the all hidden output.
/// Shape :: [seq_length, num_directions, batch_size, hidden_size]
Value allocAllHidden(
    ConversionPatternRewriter &rewriter, Location loc, Value X, Value R) {
  LLVM_DEBUG(llvm::dbgs() << "allocAllHidden\n");
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  RankedTensorType zeroType =
      RankedTensorType::get({dimAt(X, 0), 1, dimAt(X, 1), dimAt(R, 2)},
          X.getType().cast<ShapedType>().getElementType());
  DenseElementsAttr zeroAttr = DenseElementsAttr::get(zeroType, 0.0f);
  return create.onnx.constant(zeroAttr);
}

/// Allocate the hidden or cell output.
mlir::Value allocHiddenOrCell(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value X, mlir::Value W, mlir::Value R) {
  LLVM_DEBUG(llvm::dbgs() << "allocHiddenOrCell\n");
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  RankedTensorType zeroType = RankedTensorType::get(
      {/*num_directions=*/dimAt(W, 0), /*batch_size=*/dimAt(X, 1),
          /*hidden_size=*/dimAt(R, 2)},
      X.getType().cast<ShapedType>().getElementType());
  DenseElementsAttr zeroAttr = DenseElementsAttr::get(zeroType, 0.0f);
  return create.onnx.constant(zeroAttr);
}

/// Allocate the intermediate hidden or cell states.
/// Shape :: [batch_size, hidden_size]
Value allocIntermediateState(
    ConversionPatternRewriter &rewriter, Location loc, Value X, Value R) {
  LLVM_DEBUG(llvm::dbgs() << "allocIntermediateState\n");
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  RankedTensorType zeroType =
      RankedTensorType::get({/*batch_size=*/dimAt(X, 1),
                                /*hidden_size=*/dimAt(R, 2)},
          X.getType().cast<ShapedType>().getElementType());
  DenseElementsAttr zeroAttr = DenseElementsAttr::get(zeroType, 0.0f);
  return create.onnx.constant(zeroAttr);
}

/// Initialize the intermediate hidden and cell states.
/// forward(reverse)Ht, forward(reverse)Ct
void initializeIntermediateStates(ConversionPatternRewriter &rewriter,
    Location loc, Value &forwardHt, Value &reverseHt, Value &forwardCt,
    Value &reverseCt, Value initialH, Value initialC, Type elementType,
    StringRef direction, bool onlyHidden) {
  LLVM_DEBUG(llvm::dbgs() << "initializeIntermediateStates\n");
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  Value zeroIndex = create.onnx.constantInt64({0});
  Value oneIndex = create.onnx.constantInt64({1});
  Value twoIndex = create.onnx.constantInt64({2});

  Value boundVal = (direction == FORWARD || direction == BIDIRECTIONAL)
                       ? forwardHt
                       : reverseHt;
  auto valShape = boundVal.getType().cast<ShapedType>().getShape();
  RankedTensorType sliceType =
      RankedTensorType::get({1, valShape[0], valShape[1]},
          boundVal.getType().cast<RankedTensorType>().getElementType());
  RankedTensorType valType = boundVal.getType().cast<RankedTensorType>();
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    if (!isNoneValue(initialH)) {
      forwardHt = create.onnx.slice(
          sliceType, initialH, zeroIndex, oneIndex, zeroIndex, oneIndex);
      forwardHt = create.onnx.squeeze(valType, forwardHt, zeroIndex);
    }
    if (!onlyHidden && !isNoneValue(initialC)) {
      forwardCt = create.onnx.slice(
          sliceType, initialC, zeroIndex, oneIndex, zeroIndex, oneIndex);
      forwardCt = create.onnx.squeeze(valType, forwardCt, zeroIndex);
    }
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    if (!isNoneValue(initialH)) {
      if (direction == REVERSE) {
        reverseHt = create.onnx.slice(
            sliceType, initialH, zeroIndex, oneIndex, zeroIndex, oneIndex);
        reverseHt = create.onnx.squeeze(valType, reverseHt, zeroIndex);
      } else {
        reverseHt = create.onnx.slice(
            sliceType, initialH, oneIndex, twoIndex, zeroIndex, oneIndex);
        reverseHt = create.onnx.squeeze(valType, reverseHt, zeroIndex);
      }
    }
    if (!onlyHidden and !isNoneValue(initialC)) {
      if (direction == REVERSE) {
        reverseCt = create.onnx.slice(
            sliceType, initialC, zeroIndex, oneIndex, zeroIndex, oneIndex);
        reverseCt = create.onnx.squeeze(valType, reverseCt, zeroIndex);
      } else {
        reverseCt = create.onnx.slice(
            sliceType, initialC, oneIndex, twoIndex, zeroIndex, oneIndex);
        reverseCt = create.onnx.squeeze(valType, reverseCt, zeroIndex);
      }
    }
  }
}

/// Store a state into the output of the RNN op.
/// The input state is 2D and the output state is 3D with '1' or '2' is
/// pretended, depending on 'direction'.
void stateToOutputForHiddenOrCell(ConversionPatternRewriter &rewriter,
    Location loc, Value forwardVal, Value reverseVal, StringRef direction,
    Value &output) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  if (direction == FORWARD || direction == REVERSE) {
    Value val = (direction == FORWARD) ? forwardVal : reverseVal;
    output = val;
  } else { // BIDIRECTIONAL
    SmallVector<int64_t, 4> bForwardValShape(
        forwardVal.getType().cast<ShapedType>().getShape());
    SmallVector<int64_t, 4> bValShape(
        forwardVal.getType().cast<ShapedType>().getShape());
    SmallVector<int64_t, 4> bReverseValShape(
        reverseVal.getType().cast<ShapedType>().getShape());
    bForwardValShape.insert(bForwardValShape.begin(), 1);
    bReverseValShape.insert(bReverseValShape.begin(), 1);
    bValShape.insert(bValShape.begin(), 2);
    Type valElementType =
        forwardVal.getType().cast<ShapedType>().getElementType();
    Value zero = create.onnx.constantInt64({0});
    Value bForwardVal = create.onnx.unsqueeze(
        RankedTensorType::get(bForwardValShape, valElementType), forwardVal,
        zero);
    Value bReverseVal = create.onnx.unsqueeze(
        RankedTensorType::get(bReverseValShape, valElementType), reverseVal,
        zero);
    output =
        create.onnx.concat(RankedTensorType::get(bValShape, valElementType),
            {bForwardVal, bReverseVal}, 0);
  }
}

// Apply an activation function on a given scalar operand.
Value applyActivation(OpBuilder &rewriter, Location loc,
    RNNActivation activation, Value operand) {
  Value res;

  std::vector<mlir::NamedAttribute> attributes;
  if (activation.alpha) {
    attributes.emplace_back(
        rewriter.getNamedAttr("alpha", activation.alpha.value()));
  }
  if (activation.beta) {
    attributes.emplace_back(
        rewriter.getNamedAttr("beta", activation.beta.value()));
  }
  Type resType = operand.getType();

  // Change equality to be case insensitive.
  if (activation.name.equals_insensitive("relu"))
    res = rewriter.create<ONNXReluOp>(loc, resType, operand);
  else if (activation.name.equals_insensitive("tanh"))
    res = rewriter.create<ONNXTanhOp>(loc, resType, operand);
  else if (activation.name.equals_insensitive("sigmoid"))
    res = rewriter.create<ONNXSigmoidOp>(loc, resType, operand);
  else if (activation.name.equals_insensitive("affine"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_insensitive("leakyrelu"))
    res = rewriter.create<ONNXLeakyReluOp>(loc, resType, operand, attributes);
  else if (activation.name.equals_insensitive("thresholdedrelu"))
    res = rewriter.create<ONNXThresholdedReluOp>(
        loc, resType, operand, attributes);
  else if (activation.name.equals_insensitive("scaledtanh"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_insensitive("hardsigmoid"))
    res = rewriter.create<ONNXHardSigmoidOp>(loc, resType, operand, attributes);
  else if (activation.name.equals_insensitive("elu"))
    res = rewriter.create<ONNXEluOp>(loc, resType, operand, attributes);
  else if (activation.name.equals_insensitive("softsign"))
    res = rewriter.create<ONNXSoftsignOp>(loc, resType, operand);
  else if (activation.name.equals_insensitive("softplus"))
    res = rewriter.create<ONNXSoftplusOp>(loc, resType, operand);
  else
    llvm_unreachable("Unsupported activation");

  return res;
}

/// Create a copy of a slice of X at a specific timestep.
Value emitXSliceAt(ConversionPatternRewriter &rewriter, Location loc, Value X,
    Value timestepIV) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  int64_t batchSize = dimAt(X, 1);
  int64_t inputSize = dimAt(X, 2);
  Type elementType = X.getType().cast<ShapedType>().getElementType();
  RankedTensorType sliceXType =
      RankedTensorType::get({1, batchSize, inputSize}, elementType);
  RankedTensorType squeezedXType =
      RankedTensorType::get({batchSize, inputSize}, elementType);
  Value sliceX = create.onnx.slice(sliceXType, X, timestepIV,
      create.onnx.add(timestepIV, create.onnx.constantInt64({1})),
      create.onnx.constantInt64({0}), create.onnx.constantInt64({1}));
  sliceX = create.onnx.squeeze(
      squeezedXType, sliceX, create.onnx.constantInt64({0}));
  return sliceX;
}

} // namespace stablehlo

} // namespace onnx_mlir
