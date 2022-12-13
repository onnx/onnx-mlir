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
  Value X = operandAdaptor.input();
  // Get attributes.
  ArrayAttr kernelShape = poolOp.kernel_shape();
  ArrayAttr strides = poolOp.strides();
  StringRef paddingType = poolOp.padding_type();

  // Get bounds
  MemRefBoundsIndexCapture XBounds(X);
  IndexExpr B = XBounds.getDim(0);
  IndexExpr HI = XBounds.getDim(1);
  IndexExpr WI = XBounds.getDim(2);
  IndexExpr CI = XBounds.getDim(3);
  IndexExpr KH = LiteralIndexExpr(kernelShape[0].cast<IntegerAttr>().getInt());
  IndexExpr KW = LiteralIndexExpr(kernelShape[1].cast<IntegerAttr>().getInt());
  IndexExpr strideH = LiteralIndexExpr(strides[0].cast<IntegerAttr>().getInt());
  IndexExpr strideW = LiteralIndexExpr(strides[1].cast<IntegerAttr>().getInt());

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
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()))
    return success();

  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  ZHighPoolingOpShapeHelper<ZHighMaxPool2DOp> shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(
      inputType.getElementType(), inputType.getEncoding());
}

//===----------------------------------------------------------------------===//
// AvgPool2DOp
//===----------------------------------------------------------------------===//

LogicalResult ZHighAvgPool2DOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()))
    return success();

  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  ZHighPoolingOpShapeHelper<ZHighAvgPool2DOp> shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(
      inputType.getElementType(), inputType.getEncoding());
}

} // namespace zhigh
} // namespace onnx_mlir
