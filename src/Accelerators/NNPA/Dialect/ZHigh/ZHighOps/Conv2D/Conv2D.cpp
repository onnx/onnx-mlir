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

  Builder builder(getContext());
  ZHighConv2DOpAdaptor operandAdaptor(*this);
  ZHighConv2DOpShapeHelper shapeHelper(this);
  if (failed(shapeHelper.computeShape(operandAdaptor)))
    return emitError("Failed to scan ZHigh Conv2D parameters successfully");

  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.dimsForOutput(0), outputDims);
  RankedTensorType inputType = input().getType().cast<RankedTensorType>();
  updateType(getResult(), outputDims, inputType.getElementType(),
      inputType.getEncoding());
  return success();
}

} // namespace zhigh
} // namespace onnx_mlir
