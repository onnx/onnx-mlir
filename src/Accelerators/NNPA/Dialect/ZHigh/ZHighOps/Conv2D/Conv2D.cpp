/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Conv2D.cpp - ZHigh Operations ---------------------===//
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

LogicalResult ZHighConv2DOpShapeHelper::computeShape() {
  ZHighConv2DOp convOp = llvm::dyn_cast<ZHighConv2DOp>(op);
  ZHighConv2DOp::Adaptor operandAdaptor(operands);
  // Get operands.
  // X: [B, HI, WI, CI]
  Value X = operandAdaptor.input();
  // W: [KH, KW, CI, CO]
  Value W = operandAdaptor.input_kernel();
  // Get attributes.
  ArrayAttr strides = convOp.strides();
  StringRef paddingType = convOp.padding_type();

  // Get bounds
  SmallVector<IndexExpr, 4> XDims, WDims;
  createIE->getShapeAsDims(X, XDims);
  createIE->getShapeAsDims(W, WDims);
  IndexExpr B = XDims[0];
  IndexExpr HI = XDims[1];
  IndexExpr WI = XDims[2];
  IndexExpr CI = XDims[3];
  IndexExpr KH = WDims[0];
  IndexExpr KW = WDims[1];
  IndexExpr CO = WDims[3];
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

  // Output shape: [B, HO, WO, CO]
  DimsExpr outputDims;
  outputDims.emplace_back(B);
  outputDims.emplace_back(HO);
  outputDims.emplace_back(WO);
  outputDims.emplace_back(CO);

  // Keep all original dimensions.
  allOriginalDims.emplace_back(B);
  allOriginalDims.emplace_back(CI);
  allOriginalDims.emplace_back(HI);
  allOriginalDims.emplace_back(WI);
  allOriginalDims.emplace_back(CO);
  allOriginalDims.emplace_back(HO);
  allOriginalDims.emplace_back(WO);

  // Save the final results.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

LogicalResult ZHighConv2DOp::verify() {
  ZHighConv2DOpAdaptor operandAdaptor(*this);
  // Get operands.
  Value K = operandAdaptor.input_kernel();
  Value B = operandAdaptor.input_bias();

  // Verify attributes.
  // - padding_type must be SAME_PADDING or VALID_PADDING.
  StringRef paddingType = padding_type();
  if (!(paddingType.equals_insensitive("SAME_PADDING") ||
          paddingType.equals_insensitive("VALID_PADDING")))
    return failure();
  // - act_func must be ACT_NONE or ACT_RELU.
  StringRef actFunc = act_func();
  if (!(actFunc.equals_insensitive("ACT_NONE") ||
          actFunc.equals_insensitive("ACT_RELU")))
    return failure();

  // Verify bias shape.
  if (!B.getType().isa<NoneType>() && hasRankedType(B) && hasRankedType(K)) {
    int64_t channelOutB = B.getType().cast<RankedTensorType>().getShape()[0];
    int64_t channelOutK = K.getType().cast<RankedTensorType>().getShape()[3];
    if (!ShapedType::isDynamic(channelOutB) &&
        !ShapedType::isDynamic(channelOutK) && (channelOutB != channelOutK))
      return failure();
  }

  // Verify kernel shape.
  ArrayAttr kernelShape = kernel_shape();
  int64_t attrKH = kernelShape[0].cast<IntegerAttr>().getInt();
  int64_t attrKW = kernelShape[1].cast<IntegerAttr>().getInt();
  if (hasRankedType(K)) {
    int64_t KH = K.getType().cast<RankedTensorType>().getShape()[0];
    int64_t KW = K.getType().cast<RankedTensorType>().getShape()[1];
    if (!ShapedType::isDynamic(KH) && KH != attrKH)
      return failure();
    if (!ShapedType::isDynamic(KW) && KW != attrKW)
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighConv2DOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!hasRankedType(input()) || !hasRankedType(input_kernel()))
    return success();

  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  ZHighConv2DOpShapeHelper shapeHelper(getOperation());
  return shapeHelper.computeShapeAndUpdateType(
      inputType.getElementType(), inputType.getEncoding());
}

} // namespace zhigh
} // namespace onnx_mlir
