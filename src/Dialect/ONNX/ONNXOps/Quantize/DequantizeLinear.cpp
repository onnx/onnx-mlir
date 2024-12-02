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

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ONNXOps"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {

// Returns known length if ty is a non-scalar 1-D vector, otherwise
// ShapedType::kDynamic.
int64_t nonScalar1DLen(ShapedType ty) {
  if (!ty.hasRank() || ty.getRank() != 1 || ty.isDynamicDim(0))
    return ShapedType::kDynamic;
  int64_t d = ty.getDimSize(0);
  return d == 1 ? ShapedType::kDynamic
                : d; // If dim size is 1 then it's considered a scalar.
}

} // namespace

namespace onnx_mlir {

template <>
LogicalResult ONNXDequantizeLinearOpShapeHelper::computeShape() {
  ONNXDequantizeLinearOpAdaptor operandAdaptor(
      operands, op->getAttrDictionary());
  RankedTensorType xTy =
      mlir::dyn_cast<RankedTensorType>(operandAdaptor.getX().getType());
  DimsExpr outputDims;
  createIE->getShapeAsDims(operandAdaptor.getX(), outputDims);

  // Get d.
  int64_t d = nonScalar1DLen(
      mlir::cast<ShapedType>(operandAdaptor.getXScale().getType()));
  if (d == ShapedType::kDynamic &&
      !isNoneValue(operandAdaptor.getXZeroPoint())) {
    d = nonScalar1DLen(
        mlir::cast<ShapedType>(operandAdaptor.getXZeroPoint().getType()));
  }

  if (d != ShapedType::kDynamic) {
    int64_t r = xTy.getRank();
    int64_t a = operandAdaptor.getAxis();
    // Checked in verify:
    assert(-r <= a && a < r && "axis out of range");
    if (a < 0)
      a += r;
    if (!outputDims[a].isLiteral()) {
      outputDims[a] = LitIE(d);
    }
    LLVM_DEBUG(llvm::dbgs() << "literal: " << outputDims[a].getLiteral()
                            << " d = " << d << "\n");
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

  Value scale = getXScale();
  auto scaleTy = mlir::cast<ShapedType>(scale.getType());
  if (scaleTy.hasRank() && scaleTy.getRank() > 1)
    return emitOpError("x_scale must be a scalar or 1-D tensor");
  int64_t scaleLen = nonScalar1DLen(scaleTy);

  Value zero = getXZeroPoint();
  int64_t zeroLen = ShapedType::kDynamic;
  if (!isNoneValue(zero)) {
    if (auto zeroTy = mlir::dyn_cast<RankedTensorType>(zero.getType())) {
      if (zeroTy.getRank() > 1)
        return emitOpError("x_zero_point must be a scalar or 1-D tensor");
      zeroLen = nonScalar1DLen(zeroTy);
      if (auto scaleTy = mlir::dyn_cast<RankedTensorType>(scale.getType())) {
        if ((isScalar(scaleTy) && scaleLen != ShapedType::kDynamic) ||
            (zeroLen != ShapedType::kDynamic && isScalar(zeroTy)) ||
            (zeroLen != ShapedType::kDynamic &&
                scaleLen != ShapedType::kDynamic && zeroLen != scaleLen))
          return emitOpError(
              "x_scale and x_zero_point must have the same shape");
      }
    }

    // TODO: Figure out whether to introduce a variant of this check from the
    // spec ("'x_zero_point' and 'x' must have same type"). It is violated in
    // in the resnet50-v1-12-qdq model where x, x_zero_point are i8, ui8.
    //
    // if (getElementType(getX().getType()) != getElementType(zero.getType()))
    //   return emitOpError("x and x_zero_point must have the same data type");

    if (getElementType(zero.getType()).isInteger(32) && zeroLen != 0)
      if (auto values = getElementAttributeFromONNXValue(zero)) {
        WideNum zero = WideNum::widen<BType::INT32>(0);
        if (!ElementsAttrBuilder::allEqual(values, zero))
          return emitOpError("x_zero_point must be 0 for data type int32");
      }
  }

  if (scaleLen == ShapedType::kDynamic && zeroLen == ShapedType::kDynamic) {
    // Either x_scale or x_zero_point is scalar, so quantization is per-tensor /
    // per layer and axis is ignored and there is nothing more to verify, or
    // their 1-D rank is unknown and we cannot verify more until they are known.
  } else {
    // If x_scale or x_zero_point is a non-scalar 1-D tensor then quantization
    // is per-axis.
    int64_t d = scaleLen != ShapedType::kDynamic ? scaleLen : zeroLen;
    if (auto xTy = mlir::dyn_cast<RankedTensorType>(getX().getType())) {
      int64_t r = xTy.getRank();
      // axis attribute must be in the range [-r,rShapedType::kDynamic].
      int64_t a = getAxis();
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
  if (!mlir::dyn_cast<RankedTensorType>(getX().getType()))
    return success();
  Type elementType = mlir::cast<ShapedType>(getY().getType()).getElementType();
  ONNXDequantizeLinearOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXDequantizeLinearOp>;
} // namespace onnx_mlir
