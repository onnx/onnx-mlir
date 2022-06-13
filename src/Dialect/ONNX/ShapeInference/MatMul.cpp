/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ MatMul.cpp - Shape Inference for MatMul Op --------------===//
//
// This file implements shape inference for the ONNX MatMul Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

ONNXMatMulOpShapeHelper::ONNXMatMulOpShapeHelper(
    ONNXMatMulOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXMatMulOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope),
      aDims(), bDims(), aPadDims(), bPadDims() {}

ONNXMatMulOpShapeHelper::ONNXMatMulOpShapeHelper(ONNXMatMulOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXMatMulOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope),
      aDims(), bDims(), aPadDims(), bPadDims() {}

LogicalResult ONNXMatMulOpShapeHelper::computeShape(
    ONNXMatMulOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of result.
  DimsExpr outputDims;

  // Get info.
  Value A = operandAdaptor.A();
  Value B = operandAdaptor.B();
  MemRefBoundsIndexCapture ABounds(A);
  MemRefBoundsIndexCapture BBounds(B);

  // Size all the arrays to padded length.
  int paddedRank = std::max(ABounds.getRank(), BBounds.getRank());
  paddedRank = std::max(paddedRank, 2);
  aDims.resize(paddedRank);
  bDims.resize(paddedRank);
  aPadDims.resize(paddedRank, false);
  bPadDims.resize(paddedRank, false);

  // Add the dims of A. All of the aDim[0]...aDim[aRank-1] are in the
  // rightmost positions, prepended by 1s to fit the paddedRankSize. (1,1,1...
  // 1, aDim[0]...aDim[aRank-1])
  LiteralIndexExpr oneIE(1);
  int aOffset = paddedRank - ABounds.getRank();
  for (int i = 0; i < aOffset; ++i) {
    aDims[i] = oneIE;
    aPadDims[i] = true;
  }
  for (unsigned int i = 0; i < ABounds.getRank(); ++i) {
    aDims[i + aOffset] = ABounds.getDim(i);
    aPadDims[i + aOffset] = false; // Pad false even if dim is sized 1.
  }
  // for B: two cases. If bRank = 1, we pad the rightmost position. Namely we
  // get (1...,1, bDim[0], 1). We use one padding credit for the rightmost
  // position. Otherwise, when bRank>1, we only pad the leading positions.
  // Namely we get (1,1,1...,1, bDim[0],.... bDim[bRank-1])
  int bOffset = paddedRank - BBounds.getRank();
  if (BBounds.getRank() == 1) {
    bDims[paddedRank - 1] = oneIE;
    bPadDims[paddedRank - 1] = true;
    bOffset--;
  }
  for (int i = 0; i < bOffset; ++i) {
    bDims[i] = oneIE;
    bPadDims[i] = true;
  }
  for (unsigned int i = 0; i < BBounds.getRank(); ++i) {
    bDims[i + bOffset] = BBounds.getDim(i);
    bPadDims[i + bOffset] = false; // Pad false even if dim is sized 1.
  }
  assert(aDims.size() == bDims.size() && "padded A&B must have same size");

  // Fill in the output dimensions, start with the non-matmul dims.
  for (int i = 0; i < paddedRank - 2; ++i) {
    // Check for broadcast, then literals, then runtime for both.
    if (aDims[i].isLiteralAndIdenticalTo(1)) {
      // A is broadcast, use B dim.
      outputDims.emplace_back(bDims[i]);
    } else if (bDims[i].isLiteralAndIdenticalTo(1)) {
      // B is a broadcast, use A dim.
      outputDims.emplace_back(aDims[i]);
    } else if (aDims[i].isLiteral() && bDims[i].isLiteral()) {
      // No broadcast, both literals, make sure they have the same value.
      if (aDims[i].getLiteral() != bDims[i].getLiteral())
        return op->emitError("Incompatible size detected");
      outputDims.emplace_back(aDims[i]);
    } else if (aDims[i].isLiteral()) {
      // A dim is a literal; use it here for output and b, since b
      // is guaranteed not to be a broadcast (earlier tests).
      outputDims.emplace_back(aDims[i]);
      bDims[i] = aDims[i]; // Add runtime check if desired.
    } else if (bDims[i].isLiteral()) {
      // A dim is a literal; use it here for output and a, since a
      // is guaranteed not to be a broadcast (earlier tests).
      outputDims.emplace_back(bDims[i]);
      aDims[i] = bDims[i]; // Add runtime check if desired.
    } else {
      // Have no broadcast or literal, just pick a for output; add runtime
      // check if desired.
      outputDims.emplace_back(aDims[i]);
    }
  }
  // We now check get the last two dimensions: NxK times KxM.
  int aN = paddedRank - 2;
  int aK = paddedRank - 1;
  int bK = paddedRank - 2;
  int bM = paddedRank - 1;
  // And test the K dimensions.
  if (aDims[aK].isLiteral() && bDims[bK].isLiteral()) {
    if (aDims[aK].getLiteral() != bDims[bK].getLiteral())
      return op->emitError("reduction dimension must be the same");
  } else if (aDims[aK].isLiteral()) {
    // Save aK dims into bK dims, in case bK dims was runtime
    bDims[bK] = aDims[aK];
  } else if (bDims[bK].isLiteral()) {
    // Save bK dims into aK dims, in case aK dims was runtime
    aDims[aK] = bDims[bK];
  }
  // Add lower N x M dimensions if they are not padded dimensions.
  if (!aPadDims[aN])
    outputDims.emplace_back(aDims[aN]);
  if (!bPadDims[bM])
    outputDims.emplace_back(bDims[bM]);
  // For the case where both aRank == bRank == 1
  if (ABounds.getRank() == 1 && BBounds.getRank() == 1) {
    outputDims.emplace_back(oneIE);
  }
  // Save the final result.
  dimsForOutput() = outputDims;
  return success();
}

} // namespace onnx_mlir
