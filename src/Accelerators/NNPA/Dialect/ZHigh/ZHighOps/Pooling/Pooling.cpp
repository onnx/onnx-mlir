/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Pooling.cpp - ZHigh Operations --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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

template <typename OP>
LogicalResult ZHighPoolingOpShapeHelper<OP>::computeShape() {
  OP poolOp = llvm::dyn_cast<OP>(op);
  typename OP::Adaptor operandAdaptor(operands);
  // Get operands.
  // X: [B, HI, WI, CI]
  Value X = operandAdaptor.getInput();
  // Get attributes.
  ArrayAttr kernelShape = poolOp.getKernelShape();
  ArrayAttr strides = poolOp.getStrides();
  StringRef paddingType = poolOp.getPaddingType();

  // Get bounds
  SmallVector<IndexExpr, 4> XDims;
  createIE->getShapeAsDims(X, XDims);
  IndexExpr B = XDims[0];
  IndexExpr HI = XDims[1];
  IndexExpr WI = XDims[2];
  IndexExpr CI = XDims[3];
  IndexExpr KH = createIE->getIntFromArrayAsLiteral(kernelShape, 0);
  IndexExpr KW = createIE->getIntFromArrayAsLiteral(kernelShape, 1);
  IndexExpr strideH = createIE->getIntFromArrayAsLiteral(strides, 0);
  IndexExpr strideW = createIE->getIntFromArrayAsLiteral(strides, 1);

  // Compute output height and weight.
  IndexExpr HO, WO;
  if (paddingType.equals_insensitive("SAME_PADDING")) {
    HO = HI.ceilDiv(strideH);
    WO = WI.ceilDiv(strideW);
  } else if (paddingType.equals_insensitive("VALID_PADDING")) {
    IndexExpr newHI = HI - KH + 1;
    IndexExpr newWI = WI - KW + 1;
    HO = newHI.ceilDiv(strideH);
    WO = newWI.ceilDiv(strideW);
  } else {
    llvm_unreachable("Unsupported padding_type");
  }

  // Output shape: [B, HO, WO, CI]
  DimsExpr outputDims;
  outputDims.emplace_back(B);
  outputDims.emplace_back(HO);
  outputDims.emplace_back(WO);
  outputDims.emplace_back(CI);

  // Keep all original dimensions.
  allOriginalDims.emplace_back(B);
  allOriginalDims.emplace_back(CI);
  allOriginalDims.emplace_back(HI);
  allOriginalDims.emplace_back(WI);
  allOriginalDims.emplace_back(HO);
  allOriginalDims.emplace_back(WO);

  // Save the final results.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// ZHigh Shape Helper template instantiation
// Keep template instantiation at the end of the file.
//===----------------------------------------------------------------------===//

template struct ZHighPoolingOpShapeHelper<ZHighMaxPool2DOp>;
template struct ZHighPoolingOpShapeHelper<ZHighAvgPool2DOp>;

//===----------------------------------------------------------------------===//
// MaxPool2DOp
//===----------------------------------------------------------------------===//

LogicalResult ZHighMaxPool2DOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getInput()))
    return success();

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(getInput().getType());
  ZHighPoolingOpShapeHelper<ZHighMaxPool2DOp> shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(
      inputType.getElementType(), inputType.getEncoding());
}

//===----------------------------------------------------------------------===//
// AvgPool2DOp
//===----------------------------------------------------------------------===//

LogicalResult ZHighAvgPool2DOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!hasRankedType(getInput()))
    return success();

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(getInput().getType());
  ZHighPoolingOpShapeHelper<ZHighAvgPool2DOp> shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(
      inputType.getElementType(), inputType.getEncoding());
}

} // namespace zhigh
} // namespace onnx_mlir
