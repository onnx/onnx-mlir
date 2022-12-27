/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ DequantizeLinear.cpp - ONNX Operations ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect DequantizeLinear operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {

// Returns known length if ty is a non-scalar 1-D vector, otherwise -1.
int64_t nonScalar1DLen(ShapedType ty) {
  if (!ty.hasRank() || ty.getRank() != 1 || ty.isDynamicDim(0))
    return -1;
  int64_t d = ty.getDimSize(0);
  return d == 1 ? -1 : d; // If dim size is 1 then it's considered a scalar.
}

} // namespace

namespace onnx_mlir {

template <>
LogicalResult ONNXDequantizeLinearOpShapeHelper::computeShape() {
  ONNXDequantizeLinearOpAdaptor operandAdaptor(
      operands, op->getAttrDictionary());
  RankedTensorType xTy =
      operandAdaptor.x().getType().dyn_cast<RankedTensorType>();
  DimsExpr outputDims;
  createIE->getShapeAsDims(operandAdaptor.x(), outputDims);

  // Get d.
  int64_t d =
      nonScalar1DLen(operandAdaptor.x_scale().getType().cast<ShapedType>());
  if (d == -1 && !isFromNone(operandAdaptor.x_zero_point())) {
    d = nonScalar1DLen(
        operandAdaptor.x_zero_point().getType().cast<ShapedType>());
  }

  if (d != -1) {
    int64_t r = xTy.getRank();
    int64_t a = operandAdaptor.axis();
    // Checked in verify:
    assert(-r <= a && a < r && "axis out of range");
    if (a < 0)
      a += r;
    if (!outputDims[a].isLiteral()) {
      outputDims[a] = LiteralIndexExpr(d);
    }
    // Checked in verify.
    assert(outputDims[a].getLiteral() == d &&
           "x_scale and x_zero_point 1-D tensor length must match the input "
           "axis dim size");
  }

  // Get values.
  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXDequantizeLinearOp::verify() {
  // Is tensor known to be a scalar (rank 0 or rank 1 with 1 element)?
  auto isScalar = [](RankedTensorType t) -> bool {
    return t.getRank() == 0 || (t.getRank() == 1 && t.getDimSize(0) == 1);
  };

  Value scale = x_scale();
  auto scaleTy = scale.getType().cast<ShapedType>();
  if (scaleTy.hasRank() && scaleTy.getRank() > 1)
    return emitOpError("x_scale must be a scalar or 1-D tensor");
  int64_t scaleLen = nonScalar1DLen(scaleTy);

  Value zero = x_zero_point();
  int64_t zeroLen = -1;
  if (!isFromNone(zero)) {
    if (auto zeroTy = zero.getType().dyn_cast<RankedTensorType>()) {
      if (zeroTy.getRank() > 1)
        return emitOpError("x_zero_point must be a scalar or 1-D tensor");
      zeroLen = nonScalar1DLen(zeroTy);
      if (auto scaleTy = scale.getType().dyn_cast<RankedTensorType>()) {
        if ((isScalar(scaleTy) && scaleLen != -1) ||
            (zeroLen != -1 && isScalar(zeroTy)) ||
            (zeroLen != -1 && scaleLen != -1 && zeroLen != scaleLen))
          return emitOpError(
              "x_scale and x_zero_point must have the same shape");
      }
    }

    // TODO: Figure out whether to introduce a variant of this check from the
    // spec ("'x_zero_point' and 'x' must have same type"). It is violated in
    // in the resnet50-v1-12-qdq model where x, x_zero_point are i8, ui8.
    //
    // if (getElementType(x().getType()) != getElementType(zero.getType()))
    //   return emitOpError("x and x_zero_point must have the same data type");

    if (getElementType(zero.getType()).isInteger(32) && zeroLen != 0)
      if (auto values = getDenseElementAttributeFromONNXValue(zero))
        if (!values.isSplat() || !values.getSplatValue<APInt>().isZero())
          return emitOpError("x_zero_point must be 0 for data type int32");
  }

  if (scaleLen == -1 && zeroLen == -1) {
    // Either x_scale or x_zero_point is scalar, so quantization is per-tensor /
    // per layer and axis is ignored and there is nothing more to verify, or
    // their 1-D rank is unknown and we cannot verify more until they are known.
  } else {
    // If x_scale or x_zero_point is a non-scalar 1-D tensor then quantization
    // is per-axis.
    int64_t d = scaleLen != -1 ? scaleLen : zeroLen;
    if (auto xTy = x().getType().dyn_cast<RankedTensorType>()) {
      int64_t r = xTy.getRank();
      // axis attribute must be in the range [-r,r-1].
      int64_t a = axis();
      if (a < -r || a >= r)
        return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
            *this->getOperation(), "axis", a,
            onnx_mlir::Diagnostic::Range<int64_t>(-r, r - 1));
      if (a < 0)
        a += r;
      if (!xTy.isDynamicDim(a) && xTy.getDimSize(a) != d)
        return emitOpError("x_scale and x_zero_point 1-D tensor length must "
                           "match the input axis dim size");
    } else {
      // Cannot verify more until x rank is known.
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXDequantizeLinearOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  if (!x().getType().dyn_cast<RankedTensorType>())
    return success();
  Type elementType = y().getType().cast<ShapedType>().getElementType();
  ONNXDequantizeLinearOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXDequantizeLinearOp>;
} // namespace onnx_mlir
