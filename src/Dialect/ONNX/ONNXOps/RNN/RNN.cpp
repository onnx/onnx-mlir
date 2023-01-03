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

#define AEE_NEW 0

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

#if AEE_NEW
template <typename OP_TYPE>
LogicalResult ONNXGenericRNNShapeHelper<OP_TYPE>::customComputeShape(
    int gates) {
  OP_TYPE rnnOp = llvm::cast<OP_TYPE>(op);
  typename OP_TYPE::Adaptor operandAdaptor(operands, op->getAttrDictionary());

  fprintf(stderr, "hi alex, start of generic rnn\n");
  Value X = operandAdaptor.X();
  Value W = operandAdaptor.W();
  Value R = operandAdaptor.R();
  bool batchwiseLayout = operandAdaptor.layout() == 1;
  fprintf(stderr, "hi alex, batchwise %d\n", (int)batchwiseLayout);

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
  if (operandAdaptor.hidden_size().has_value()) {
    hiddenSize = LiteralIndexExpr(operandAdaptor.hidden_size().value());
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
      rnnOp.hidden_sizeAttr(hiddenSizeAttr);
    }
  }

  // Get direction.
  IndexExpr numDir;
  if ((operandAdaptor.direction() == "forward") ||
      (operandAdaptor.direction() == "reverse"))
    numDir = LiteralIndexExpr(1);
  else if (operandAdaptor.direction() == "bidirectional")
    numDir = LiteralIndexExpr(2);
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
  Type yTy = op->getResult(0).getType();
  if (!yTy.isa<NoneType>()) {
    if (batchwiseLayout) {
      yOutputDims = {batchSize, seqLength, numDir, hiddenSize};
    } else {
      yOutputDims = {seqLength, numDir, batchSize, hiddenSize};
    }
    fprintf(stderr, "hi alex, set0 of generic rnn\n");
    setOutputDims(yOutputDims, 0);
    fprintf(stderr, "hi alex, set0 end of generic rnn\n");
  }

  // Y_h :: [batch_size, num_dir, hidden_size] if batchwiseLayout
  // Y_h :: [num_dir, batch_size, hidden_size] otherwise
  DimsExpr yHOutputDims;
  Type yhTy = op->getResult(1).getType();
  if (!yhTy.isa<NoneType>()) {
    if (batchwiseLayout) {
      yHOutputDims = {batchSize, numDir, hiddenSize};
    } else {
      yHOutputDims = {numDir, batchSize, hiddenSize};
    }
    fprintf(stderr, "hi alex, set1 of generic rnn\n");
    setOutputDims(yHOutputDims, 1);
    fprintf(stderr, "hi alex, set1 end of generic rnn\n");
  }

  if (op->getNumResults() == 3) {
    // Y_c :: [batch_size, num_dir, hidden_size] if batchwiseLayout
    // Y_c :: [num_dir, batch_size, hidden_size] otherwise
    DimsExpr yCOutputDims;
    Type ycTy = op->getResult(2).getType();
    if (!ycTy.isa<NoneType>()) {
      if (batchwiseLayout) {
        yCOutputDims = {batchSize, numDir, hiddenSize};
      } else {
        yCOutputDims = {numDir, batchSize, hiddenSize};
      }
    }
    fprintf(stderr, "hi alex, set2 of generic rnn\n");
    setOutputDims(yCOutputDims, 2);
    fprintf(stderr, "hi alex, set2 end of generic rnn\n");
  }
  fprintf(stderr, "hi alex, end of generic rnn\n");

  return success();
}

#else
  bool batchwiseLayout = op->layout() == 1;
  fprintf(stderr, "hi alex, batchwise %d\n", (int)batchwiseLayout);

  Value X = op->X();
  Value W = op->W();
  Value R = op->R();

  if (!X.getType().isa<RankedTensorType>() ||
      !W.getType().isa<RankedTensorType>() ||
      !R.getType().isa<RankedTensorType>()) {
    return success();
  }

  auto xTy = X.getType().cast<RankedTensorType>();
  Type elementType = xTy.getElementType();

  // xShape :: [batch_size, seq_length, input_size] if batchwiseLayout
  // xShape :: [seq_length, batch_size, input_size] otherwise
  auto xShape = xTy.getShape();
  // wShape :: [num_dir, gates*hidden_size, input_size]
  auto wShape = W.getType().cast<RankedTensorType>().getShape();
  // rShape :: [num_dir, gates*hidden_size, hidden_size]
  auto rShape = R.getType().cast<RankedTensorType>().getShape();

  if (xShape.size() != 3) {
    return op->emitError("The first input tensor must have rank 3");
  }
  if (wShape.size() != 3) {
    return op->emitError("The second input tensor must have rank 3");
  }
  if (rShape.size() != 3) {
    return op->emitError("The third input tensor must have rank 3");
  }

  // Get sequence length, batch size.
  int64_t seqLength = batchwiseLayout ? xShape[1] : xShape[0];
  int64_t batchSize = batchwiseLayout ? xShape[0] : xShape[1];

  // Get hidden size from hidden_size attribute.
  int64_t hiddenSize = -1;
  if (op->hidden_size().has_value()) {
    hiddenSize = op->hidden_size().value();
  } else {
    // Infer hidden_size from wShape and rShape if possible.
    if (!ShapedType::isDynamic(rShape[2]))
      hiddenSize = rShape[2];
    else if (!ShapedType::isDynamic(rShape[1]))
      hiddenSize = rShape[1] / gates;
    else if (!ShapedType::isDynamic(wShape[1]))
      hiddenSize = wShape[1] / gates;
    // Update hidden_size attribute.
    if (!ShapedType::isDynamic(hiddenSize)) {
      auto builder = Builder(op->getContext());
      auto hiddenSizeAttr =
          IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
              APInt(64, /*value=*/hiddenSize, /*isSigned=*/true));
      op->hidden_sizeAttr(hiddenSizeAttr);
    }
  }
  // Get direction.
  int64_t numDir;
  if ((op->direction() == "forward") || (op->direction() == "reverse"))
    numDir = 1;
  else if (op->direction() == "bidirectional")
    numDir = 2;
  else
    return op->emitError(
        "direction attribute must be one of the strings: forward, "
        "reverse, and bidirectional");

  // Set result types. There are always 2 (RNN, GRU) or 3 results
  // but they are sometimes optional in which case they have NoneType.
  assert((op->getNumResults() == 2 || op->getNumResults() == 3) &&
         "RNN, GRU have 2 results, LSTM has 3");
  // Y :: [batch_size, seq_length, num_dir, hidden_size] if batchwiseLayout
  // Y :: [seq_length, num_dir, batch_size, hidden_size] otherwise
  Type yTy = op->getResult(0).getType();
  if (!yTy.isa<NoneType>()) {
    if (batchwiseLayout) {
      yTy = RankedTensorType::get(
          {batchSize, seqLength, numDir, hiddenSize}, elementType);
    } else {
      yTy = RankedTensorType::get(
          {seqLength, numDir, batchSize, hiddenSize}, elementType);
    }
    op->getResult(0).setType(yTy);
  }
  // Y_h :: [batch_size, num_dir, hidden_size] if batchwiseLayout
  // Y_h :: [num_dir, batch_size, hidden_size] otherwise
  Type yhTy = op->getResult(1).getType();
  if (!yhTy.isa<NoneType>()) {
    if (batchwiseLayout) {
      yhTy =
          RankedTensorType::get({batchSize, numDir, hiddenSize}, elementType);
    } else {
      yhTy =
          RankedTensorType::get({numDir, batchSize, hiddenSize}, elementType);
    }
    op->getResult(1).setType(yhTy);
  }
  if (op->getNumResults() == 3) {
    // Y_c :: [batch_size, num_dir, hidden_size] if batchwiseLayout
    // Y_c :: [num_dir, batch_size, hidden_size] otherwise
    Type ycTy = op->getResult(2).getType();
    if (!ycTy.isa<NoneType>()) {
      if (batchwiseLayout) {
        ycTy =
            RankedTensorType::get({batchSize, numDir, hiddenSize}, elementType);
      } else {
        ycTy =
            RankedTensorType::get({numDir, batchSize, hiddenSize}, elementType);
      }
      op->getResult(2).setType(ycTy);
    }
  }
  return success();
}
#endif

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
#if AEE_NEW
  if (!X().getType().isa<RankedTensorType>() ||
      !W().getType().isa<RankedTensorType>() ||
      !R().getType().isa<RankedTensorType>()) {
    return success();
  }
  Type elementType = X().getType().cast<RankedTensorType>().getElementType();
  ONNXGRUOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
#else
  int gates = 3;
  return RNNShapeInference(this, gates);
#endif
}

//===----------------------------------------------------------------------===//
// LSTM
//===----------------------------------------------------------------------===//

LogicalResult ONNXLSTMOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
#if AEE_NEW
  if (!X().getType().isa<RankedTensorType>() ||
      !W().getType().isa<RankedTensorType>() ||
      !R().getType().isa<RankedTensorType>()) {
    return success();
  }
  Type elementType = X().getType().cast<RankedTensorType>().getElementType();
  ONNXLSTMOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
#else
  int gates = 4;
  return RNNShapeInference(this, gates);
#endif
}

//===----------------------------------------------------------------------===//
// RNN
//===----------------------------------------------------------------------===//

LogicalResult ONNXRNNOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
#if AEE_NEW
  if (!X().getType().isa<RankedTensorType>() ||
      !W().getType().isa<RankedTensorType>() ||
      !R().getType().isa<RankedTensorType>()) {
    return success();
  }
  Type elementType = X().getType().cast<RankedTensorType>().getElementType();
  ONNXRNNOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
#else
  int gates = 1;
  return RNNShapeInference(this, gates);
#endif
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXGenericRNNShapeHelper<mlir::ONNXGRUOp>;
template struct ONNXGenericRNNShapeHelper<mlir::ONNXLSTMOp>;
template struct ONNXGenericRNNShapeHelper<mlir::ONNXRNNOp>;
} // namespace onnx_mlir
