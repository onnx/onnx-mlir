/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- SDPA3D.cpp - ZHigh Operations -------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighSDPA3DOpShapeHelper::computeShape() {
  ZHighSDPA3DOp fusedOp =
      llvm::dyn_cast<ZHighSDPA3DOp>(op);
  ZHighSDPA3DOp::Adaptor operandAdaptor(operands, fusedOp);
  DimsExpr outputDims;

  Value Q = operandAdaptor.getQ();
  Value KT = operandAdaptor.getKT();
  Value V = operandAdaptor.getV();

  SmallVector<IndexExpr, 4> QDims, KTDims, VDims;
  createIE->getShapeAsDims(Q, QDims);
  createIE->getShapeAsDims(KT, KTDims);
  createIE->getShapeAsDims(V, VDims);

  if (QDims.size() != 3 || KTDims.size() != 3 || VDims.size() != 3)
    return failure();

  // transposeKT=0: KT: [s, n, p] -> P = KTDims[2]
  // transposeKT=1: K:  [s, p, n] -> P = KTDims[1]
  int64_t transposeKT = fusedOp.getTransposeKT();
  IndexExpr S = QDims[0];
  IndexExpr M = QDims[1];
  IndexExpr N = QDims[2];
  IndexExpr P = (transposeKT != 0) ? KTDims[1] : KTDims[2];
  IndexExpr QQ = VDims[2];

  outputDims.emplace_back(S);
  outputDims.emplace_back(M);
  outputDims.emplace_back(QQ);

  // Keep all original dimensions: s, m, n, p, q
  allOriginalDims.emplace_back(S);
  allOriginalDims.emplace_back(M);
  allOriginalDims.emplace_back(N);
  allOriginalDims.emplace_back(P);
  allOriginalDims.emplace_back(QQ);

  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighSDPA3DOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getQ()) || !hasRankedType(getKT()) ||
      !hasRankedType(getV()))
    return success();

  ZHighSDPA3DOpShapeHelper shapeHelper(getOperation());
  shapeHelper.computeShapeAndAssertOnFailure();

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.getOutputDims(), outputDims);
  Type elementType =
      mlir::cast<ShapedType>(getResult().getType()).getElementType();
  ZTensorEncodingAttr encoding = ZTensorEncodingAttr::get(
      this->getContext(), ZTensorEncodingAttr::DataLayout::_3DS);

  updateType(getOperation(), getResult(), outputDims, elementType, encoding);
  return success();
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

LogicalResult ZHighSDPA3DOp::verify() {
  ZHighSDPA3DOpAdaptor operandAdaptor(*this);
  Value Q = operandAdaptor.getQ();
  Value KT = operandAdaptor.getKT();
  Value Mask = operandAdaptor.getMask();
  Value V = operandAdaptor.getV();

  ZTensorEncodingAttr::DataLayout qLayout = getZTensorLayout(Q.getType());
  ZTensorEncodingAttr::DataLayout ktLayout = getZTensorLayout(KT.getType());
  ZTensorEncodingAttr::DataLayout maskLayout = getZTensorLayout(Mask.getType());
  ZTensorEncodingAttr::DataLayout vLayout = getZTensorLayout(V.getType());

  if (qLayout != ZTensorEncodingAttr::DataLayout::_3DS)
    return emitOpError("Q must be 3DS layout");
  if (ktLayout != ZTensorEncodingAttr::DataLayout::_3DS)
    return emitOpError("KT must be 3DS layout");
  if (maskLayout != ZTensorEncodingAttr::DataLayout::_3DS)
    return emitOpError("Mask must be 3DS layout");
  if (vLayout != ZTensorEncodingAttr::DataLayout::_3DS)
    return emitOpError("V must be 3DS layout");

  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
