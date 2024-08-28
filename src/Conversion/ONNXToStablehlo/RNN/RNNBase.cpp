/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- RNNBase.cpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2023-2024
//
// =============================================================================
//
// This file defines base functions for lowering the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/RNN/RNNBase.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace stablehlo {

/// Allocate the all hidden output.
/// Shape :: [seq_length, num_directions, batch_size, hidden_size]
Value allocAllHidden(
    ConversionPatternRewriter &rewriter, Location loc, Value X, Value R) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  RankedTensorType zeroType =
      RankedTensorType::get({dimAt(X, 0), 1, dimAt(X, 1), dimAt(R, 2)},
          mlir::cast<ShapedType>(X.getType()).getElementType());
  DenseElementsAttr zeroAttr = DenseElementsAttr::get(zeroType, 0.0f);
  return create.onnx.constant(zeroAttr);
}

/// Allocate the hidden or cell output.
Value allocHiddenOrCell(mlir::ConversionPatternRewriter &rewriter, Location loc,
    Value X, Value W, Value R) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  RankedTensorType zeroType = RankedTensorType::get(
      {/*num_directions=*/dimAt(W, 0), /*batch_size=*/dimAt(X, 1),
          /*hidden_size=*/dimAt(R, 2)},
      mlir::cast<ShapedType>(X.getType()).getElementType());
  DenseElementsAttr zeroAttr = DenseElementsAttr::get(zeroType, 0.0f);
  return create.onnx.constant(zeroAttr);
}

/// Allocate the intermediate hidden or cell states.
/// Shape :: [batch_size, hidden_size]
Value allocIntermediateState(
    ConversionPatternRewriter &rewriter, Location loc, Value X, Value R) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  RankedTensorType zeroType =
      RankedTensorType::get({/*batch_size=*/dimAt(X, 1),
                                /*hidden_size=*/dimAt(R, 2)},
          mlir::cast<ShapedType>(X.getType()).getElementType());
  DenseElementsAttr zeroAttr = DenseElementsAttr::get(zeroType, 0.0f);
  return create.onnx.constant(zeroAttr);
}

/// Initialize the intermediate hidden and cell states.
/// forward(reverse)Ht, forward(reverse)Ct
void initializeIntermediateStates(ConversionPatternRewriter &rewriter,
    Location loc, Value &forwardHt, Value &reverseHt, Value &forwardCt,
    Value &reverseCt, Value initialH, Value initialC, Type elementType,
    StringRef direction, bool onlyHidden) {
  MultiDialectBuilder<OnnxBuilder, StablehloBuilder> create(rewriter, loc);

  Value zeroIndex = create.stablehlo.constantI64(0);
  Value zeroIndex1D = create.onnx.constantInt64({0});
  Value oneIndex = create.stablehlo.constantI64(1);

  Value boundVal = (direction == FORWARD || direction == BIDIRECTIONAL)
                       ? forwardHt
                       : reverseHt;
  auto valShape = mlir::cast<ShapedType>(boundVal.getType()).getShape();
  SmallVector<int64_t> sliceSizes = {1, valShape[0], valShape[1]};
  SmallVector<Value> firstStartIndices = {zeroIndex, zeroIndex, zeroIndex};
  SmallVector<Value> secondStartIndices = {oneIndex, zeroIndex, zeroIndex};

  RankedTensorType valType = mlir::cast<RankedTensorType>(boundVal.getType());
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    if (!isNoneValue(initialH)) {
      forwardHt = create.stablehlo.dynamic_slice(
          initialH, firstStartIndices, sliceSizes);
      forwardHt = create.onnx.squeeze(valType, forwardHt, zeroIndex1D);
    }
    if (!onlyHidden && !isNoneValue(initialC)) {
      forwardCt = create.stablehlo.dynamic_slice(
          initialC, firstStartIndices, sliceSizes);
      forwardCt = create.onnx.squeeze(valType, forwardCt, zeroIndex1D);
    }
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    if (!isNoneValue(initialH)) {
      if (direction == REVERSE) {
        reverseHt = create.stablehlo.dynamic_slice(
            initialH, firstStartIndices, sliceSizes);
        reverseHt = create.onnx.squeeze(valType, reverseHt, zeroIndex1D);
      } else {
        reverseHt = create.stablehlo.dynamic_slice(
            initialH, secondStartIndices, sliceSizes);
        reverseHt = create.onnx.squeeze(valType, reverseHt, zeroIndex1D);
      }
    }
    if (!onlyHidden and !isNoneValue(initialC)) {
      if (direction == REVERSE) {
        reverseCt = create.stablehlo.dynamic_slice(
            initialC, firstStartIndices, sliceSizes);
        reverseCt = create.onnx.squeeze(valType, reverseCt, zeroIndex1D);
      } else {
        reverseCt = create.stablehlo.dynamic_slice(
            initialC, secondStartIndices, sliceSizes);
        reverseCt = create.onnx.squeeze(valType, reverseCt, zeroIndex1D);
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
        mlir::cast<ShapedType>(forwardVal.getType()).getShape());
    SmallVector<int64_t, 4> bValShape(
        mlir::cast<ShapedType>(forwardVal.getType()).getShape());
    SmallVector<int64_t, 4> bReverseValShape(
        mlir::cast<ShapedType>(reverseVal.getType()).getShape());
    bForwardValShape.insert(bForwardValShape.begin(), 1);
    bReverseValShape.insert(bReverseValShape.begin(), 1);
    bValShape.insert(bValShape.begin(), 2);
    Type valElementType =
        mlir::cast<ShapedType>(forwardVal.getType()).getElementType();
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

/// Create a copy of a slice of X at a specific timestep.
Value emitXSliceAt(ConversionPatternRewriter &rewriter, Location loc, Value X,
    Value timestepIV) {
  MultiDialectBuilder<OnnxBuilder, StablehloBuilder> create(rewriter, loc);
  int64_t batchSize = dimAt(X, 1);
  int64_t inputSize = dimAt(X, 2);
  Type elementType = mlir::cast<ShapedType>(X.getType()).getElementType();
  RankedTensorType squeezedXType =
      RankedTensorType::get({batchSize, inputSize}, elementType);
  SmallVector<int64_t> sliceSizes = {1, batchSize, inputSize};
  Value zeroIndex = create.stablehlo.constantI64(0);
  Value timestepIV0D = create.stablehlo.reshape(
      RankedTensorType::get({}, rewriter.getI64Type()), timestepIV);
  SmallVector<Value> startIndices = {timestepIV0D, zeroIndex, zeroIndex};
  Value sliceX = create.stablehlo.dynamic_slice(X, startIndices, sliceSizes);
  sliceX = create.onnx.squeeze(
      squeezedXType, sliceX, create.onnx.constantInt64({0}));
  return sliceX;
}

} // namespace stablehlo

} // namespace onnx_mlir
