/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Pad.cpp - ONNX Operations -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Pad operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

ONNXPadOpShapeHelper::ONNXPadOpShapeHelper(
    ONNXPadOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXPadOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope),
      pads() {}

ONNXPadOpShapeHelper::ONNXPadOpShapeHelper(ONNXPadOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXPadOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope),
      pads() {}

LogicalResult ONNXPadOpShapeHelper::computeShape(
    ONNXPadOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
  uint64_t dataRank = dataBounds.getRank();

  // Initialize context and results (pads & output)
  pads.resize(2 * dataRank); // pads two sides of each axis.
  outputDims.resize(dataRank);

  // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
  // where
  // - xi_begin: the number of pad values added at the beginning of axis `i`
  // - xi_end: the number of pad values added at the end of axis `i`.
  ArrayValueIndexCapture padsCapture(
      operandAdaptor.pads(), fGetDenseVal, fLoadVal);

  // Calculate output dimension sizes.
  for (uint64_t i = 0; i < dataRank; i++) {
    // Get begin/end pads.
    SymbolIndexExpr padBegin(padsCapture.getSymbol(i));
    SymbolIndexExpr padEnd(padsCapture.getSymbol(i + dataRank));
    if (padBegin.isUndefined() || padEnd.isUndefined())
      return op->emitError("pad parameter could not be processed");
    // Get input dim.
    DimIndexExpr dimInput(dataBounds.getDim(i));

    // Calculation for output size.
    IndexExpr dimOutputFinal = padBegin + dimInput + padEnd;

    // Save results.
    pads[i] = padBegin;
    pads[i + dataRank] = padEnd;
    outputDims[i] = dimOutputFinal;
  }

  // Save the final result.
  setOutputDims(outputDims);

  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::verify() {
  ShapedType dataTy = data().getType().cast<ShapedType>();
  Type constTy = constant_value().getType();

  if (!constTy.isa<NoneType>()) {
    // Check that the constant has the same element type as the input
    ShapedType shapedConstTy = constTy.cast<ShapedType>();
    if (dataTy.getElementType() != shapedConstTy.getElementType()) {
      return emitOpError("Pad with constant_value that doesn't match the "
                         "element type of the input.");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXPadOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!data().getType().isa<RankedTensorType>() ||
      !pads().getType().isa<RankedTensorType>())
    return success();

  auto elementType = data().getType().cast<ShapedType>().getElementType();
  return shapeHelperInferShapes<ONNXPadOpShapeHelper, ONNXPadOp,
      ONNXPadOpAdaptor>(*this, elementType);
}
