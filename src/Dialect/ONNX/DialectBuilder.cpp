/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- DialectBuilder.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains builder functions for ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;

// Identity affine
namespace onnx_mlir {

//====-------------------------- ONNX Builder ---------------------------===//

Value OnnxBuilder::add(Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return b.create<ONNXAddOp>(loc, toTensor(A), toTensor(B));
}

Value OnnxBuilder::cast(Value input, TypeAttr to) const {
  Type resultType;
  if (input.getType().cast<ShapedType>().hasRank())
    resultType = RankedTensorType::get(
        input.getType().cast<ShapedType>().getShape(), to.getValue());
  else
    resultType = UnrankedTensorType::get(to.getValue());
  return b.create<ONNXCastOp>(loc, resultType, input, to);
}

Value OnnxBuilder::ceil(Value input) const {
  return b.create<ONNXCeilOp>(loc, toTensor(input.getType()), input);
}

Value OnnxBuilder::constant(Attribute denseAttr) const {
  return b.create<ONNXConstantOp>(loc, Attribute(), denseAttr);
}

Value OnnxBuilder::constantFromRawBuffer(Type resultType, char *buf) const {
  DenseElementsAttr denseAttr =
      createDenseElementsAttrFromRawBuffer(resultType, buf);
  return b.create<ONNXConstantOp>(loc, resultType, Attribute(), denseAttr,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
      ArrayAttr());
}

Value OnnxBuilder::div(Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return b.create<ONNXDivOp>(loc, toTensor(A), toTensor(B));
}

Value OnnxBuilder::matmul(Type Y, Value A, Value B, bool useGemm) const {
  // Gemm only supports rank 2.
  bool canUseGemm = useGemm && A.getType().isa<ShapedType>() &&
                    A.getType().cast<ShapedType>().hasRank() &&
                    (A.getType().cast<ShapedType>().getRank() == 2) &&
                    B.getType().isa<ShapedType>() &&
                    B.getType().cast<ShapedType>().hasRank() &&
                    (B.getType().cast<ShapedType>().getRank() == 2);
  auto aValue = toTensor(A);
  auto bValue = toTensor(B);
  if (canUseGemm)
    return b.create<ONNXGemmOp>(loc, Y, aValue, bValue,
        b.createOrFold<ONNXNoneOp>(loc),
        /*alpha=*/b.getF32FloatAttr(1.0), /*beta=*/b.getF32FloatAttr(1.0),
        /*transA=*/
        IntegerAttr::get(b.getIntegerType(64, /*isSigned=*/true),
            APInt(64, 0, /*isSigned=*/true)),
        /*transB=*/
        IntegerAttr::get(b.getIntegerType(64, /*isSigned=*/true),
            APInt(64, 0, /*isSigned=*/true)));
  return b.create<ONNXMatMulOp>(loc, toTensor(Y), aValue, bValue);
}

Value OnnxBuilder::min(ValueRange inputs) const {
  assert(inputs.size() >= 2 && "Expect at least two inputs");
  Type elementType = inputs[0].getType().cast<ShapedType>().getElementType();
  assert(llvm::all_of(inputs, [elementType](Value v) {
    return (v.getType().cast<ShapedType>().getElementType() == elementType);
  }) && "All inputs must have the same element type");
  Type outputType = inputs[0].getType();
  for (uint64_t i = 1; i < inputs.size(); ++i)
    outputType = OpTrait::util::getBroadcastedType(
        toTensor(outputType), inputs[i].getType());
  return b.create<ONNXMinOp>(loc, toTensor(outputType), inputs);
}

Value OnnxBuilder::mul(Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return b.create<ONNXMulOp>(loc, toTensor(A), toTensor(B));
}

Value OnnxBuilder::reduceSum(Type outputType, Value data, Value axes,
    bool keepdims, bool noop_with_empty_axes) const {
  int64_t i_keepdims = keepdims; // 0 if false, 1 if true
  int64_t i_noop_with_empty_axes = noop_with_empty_axes; // ditto
  return b.create<ONNXReduceSumOp>(loc, toTensor(outputType), toTensor(data),
      toTensor(axes), i_keepdims, i_noop_with_empty_axes);
}

Value OnnxBuilder::reshape(Type outputType, Value input, Value shape) const {
  return b.create<ONNXReshapeOp>(
      loc, toTensor(outputType), toTensor(input), toTensor(shape));
}

Value OnnxBuilder::squeeze(Type outputType, Value data, Value axes) const {
  return b.create<ONNXSqueezeOp>(
      loc, toTensor(outputType), toTensor(data), toTensor(axes));
}

Value OnnxBuilder::sub(Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return b.create<ONNXSubOp>(loc, toTensor(A), toTensor(B));
}

Value OnnxBuilder::transpose(
    Type outputType, Value input, ArrayAttr perm) const {
  return b.create<ONNXTransposeOp>(
      loc, toTensor(outputType), toTensor(input), perm);
}

Value OnnxBuilder::toTensor(Value input) const {
  if (input.getType().isa<TensorType>())
    return input;
  assert(input.getType().isa<MemRefType>() &&
         "expect RankedMemref type when not a TensorType");
  auto aTensorTy = toTensor(input.getType());
  return b.create<UnrealizedConversionCastOp>(loc, aTensorTy, input)
      .getResult(0);
}

Type OnnxBuilder::toTensor(Type input) const {
  if (input.isa<TensorType>())
    return input;
  assert(input.isa<MemRefType>() &&
         "expect RankedMemref type when not a TensorType");
  auto aTy = input.cast<ShapedType>();
  mlir::Type elementTy = aTy.getElementType();
  if (elementTy.isa<IndexType>()) {
    elementTy = b.getIntegerType(64);
  }
  auto aTensorTy = RankedTensorType::get(aTy.getShape(), elementTy);
  return aTensorTy;
}

Value OnnxBuilder::toMemref(Value input) const {
  if (input.getType().isa<MemRefType>())
    return input;
  assert(input.getType().isa<RankedTensorType>() &&
         "expect RankedMemref type when not a TensorType");
  auto aTy = input.getType().cast<ShapedType>();
  auto aTensorTy = MemRefType::get(aTy.getShape(), aTy.getElementType());
  return b.create<UnrealizedConversionCastOp>(loc, aTensorTy, input)
      .getResult(0);
}

Value OnnxBuilder::unsqueeze(Type outputType, Value data, Value axes) const {
  return b.create<ONNXUnsqueezeOp>(
      loc, toTensor(outputType), toTensor(data), toTensor(axes));
}

Value OnnxBuilder::where(
    Type outputType, Value condition, Value X, Value Y) const {
  return b.create<ONNXWhereOp>(
      loc, toTensor(outputType), toTensor(condition), toTensor(X), toTensor(Y));
}

} // namespace onnx_mlir
