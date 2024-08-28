/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ RNN.cpp - ONNX Operations ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect RNN operations.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <typename OP_TYPE>
LogicalResult ONNXGenericRNNShapeHelper<OP_TYPE>::customComputeShape(
    int gates) {
  OP_TYPE rnnOp = llvm::cast<OP_TYPE>(op);
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());

  Value X = operandAdaptor.getX();
  Value W = operandAdaptor.getW();
  Value R = operandAdaptor.getR();
  bool batchwiseLayout = operandAdaptor.getLayout() == 1;

  // xShape :: [batch_size, seq_length, input_size] if batchwiseLayout
  // xShape :: [seq_length, batch_size, input_size] otherwise
  DimsExpr xDims, wDims, rDims;
  createIE->getShapeAsDims(X, xDims);
  // wShape :: [num_dir, gates*hidden_size, input_size]
  createIE->getShapeAsDims(W, wDims);
  // rShape :: [num_dir, gates*hidden_size, hidden_size]
  createIE->getShapeAsDims(R, rDims);

  if (xDims.size() != 3)
    return op->emitError("The first input tensor must have rank 3");
  if (wDims.size() != 3)
    return op->emitError("The second input tensor must have rank 3");
  if (rDims.size() != 3)
    return op->emitError("The third input tensor must have rank 3");

  // Get sequence length, batch size.
  IndexExpr seqLength = batchwiseLayout ? xDims[1] : xDims[0];
  IndexExpr batchSize = batchwiseLayout ? xDims[0] : xDims[1];

  // Get hidden size from hidden_size attribute.
  IndexExpr hiddenSize;
  if (operandAdaptor.getHiddenSize().has_value()) {
    hiddenSize = LitIE(operandAdaptor.getHiddenSize().value());
  } else {
    // Infer hidden_size from wShape and rShape if possible.
    if (rDims[2].isLiteral())
      hiddenSize = rDims[2];
    else if (rDims[1].isLiteral())
      hiddenSize = rDims[1].floorDiv(gates);
    else if (wDims[1].isLiteral())
      hiddenSize = wDims[1].floorDiv(gates);
    else
      // Pick one of the option above.
      hiddenSize = rDims[2];
    // Update hidden_size attribute.
    if (hiddenSize.isLiteral()) {
      auto builder = Builder(op->getContext());
      auto hiddenSizeAttr =
          IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
              APInt(64, /*value=*/hiddenSize.getLiteral(), /*isSigned=*/true));
      rnnOp.setHiddenSizeAttr(hiddenSizeAttr);
    }
  }

  // Get direction.
  IndexExpr numDir;
  if ((operandAdaptor.getDirection() == "forward") ||
      (operandAdaptor.getDirection() == "reverse"))
    numDir = LitIE(1);
  else if (operandAdaptor.getDirection() == "bidirectional")
    numDir = LitIE(2);
  else
    return op->emitError(
        "direction attribute must be one of the strings: forward, "
        "reverse, and bidirectional");

  // Set result types. There are always 2 (RNN, GRU) or 3 results
  // but they are sometimes optional in which case they have NoneType.
  assert((rnnOp->getNumResults() == 2 || rnnOp->getNumResults() == 3) &&
         "RNN, GRU have 2 results, LSTM has 3");

  // Y :: [batch_size, seq_length, num_dir, hidden_size] if batchwiseLayout
  // Y :: [seq_length, num_dir, batch_size, hidden_size] otherwise
  DimsExpr yOutputDims;
  if (!isNoneValue(op->getResult(0))) {
    if (batchwiseLayout) {
      yOutputDims = {batchSize, seqLength, numDir, hiddenSize};
    } else {
      yOutputDims = {seqLength, numDir, batchSize, hiddenSize};
    }
    setOutputDims(yOutputDims, 0);
  }

  // Y_h :: [batch_size, num_dir, hidden_size] if batchwiseLayout
  // Y_h :: [num_dir, batch_size, hidden_size] otherwise
  DimsExpr yHOutputDims;
  if (!isNoneValue(op->getResult(1))) {
    if (batchwiseLayout) {
      yHOutputDims = {batchSize, numDir, hiddenSize};
    } else {
      yHOutputDims = {numDir, batchSize, hiddenSize};
    }
    setOutputDims(yHOutputDims, 1);
  }

  if (op->getNumResults() == 3) {
    // Y_c :: [batch_size, num_dir, hidden_size] if batchwiseLayout
    // Y_c :: [num_dir, batch_size, hidden_size] otherwise
    DimsExpr yCOutputDims;
    if (!isNoneValue(op->getResult(2))) {
      if (batchwiseLayout) {
        yCOutputDims = {batchSize, numDir, hiddenSize};
      } else {
        yCOutputDims = {numDir, batchSize, hiddenSize};
      }
    }
    setOutputDims(yCOutputDims, 2);
  }

  return success();
}

template <>
mlir::LogicalResult ONNXGRUOpShapeHelper::computeShape() {
  int gates = 3;
  return customComputeShape(gates);
}

template <>
mlir::LogicalResult ONNXLSTMOpShapeHelper::computeShape() {
  int gates = 4;
  return customComputeShape(gates);
}

template <>
mlir::LogicalResult ONNXRNNOpShapeHelper::computeShape() {
  int gates = 1;
  return customComputeShape(gates);
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// GRU
//===----------------------------------------------------------------------===//

LogicalResult ONNXGRUOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW()) ||
      !hasShapeAndRank(getR())) {
    return success();
  }
  Type elementType =
      mlir::cast<RankedTensorType>(getX().getType()).getElementType();
  ONNXGRUOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// LSTM
//===----------------------------------------------------------------------===//

LogicalResult ONNXLSTMOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW()) ||
      !hasShapeAndRank(getR())) {
    return success();
  }
  Type elementType =
      mlir::cast<RankedTensorType>(getX().getType()).getElementType();
  ONNXLSTMOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// RNN
//===----------------------------------------------------------------------===//

LogicalResult ONNXRNNOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW()) ||
      !hasShapeAndRank(getR())) {
    return success();
  }
  Type elementType =
      mlir::cast<RankedTensorType>(getX().getType()).getElementType();
  ONNXRNNOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXGenericRNNShapeHelper<mlir::ONNXGRUOp>;
template struct ONNXGenericRNNShapeHelper<mlir::ONNXLSTMOp>;
template struct ONNXGenericRNNShapeHelper<mlir::ONNXRNNOp>;
} // namespace onnx_mlir
