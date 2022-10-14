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

// =============================================================================
// Basic operations
// =============================================================================

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

Value OnnxBuilder::concat(
    Type outputType, ValueRange inputs, int64_t axis) const {
  IntegerAttr concatAxisAttr =
      IntegerAttr::get(b.getIntegerType(64, /*isSigned=*/true),
          APInt(64, axis, /*isSigned=*/true));
  return b.create<ONNXConcatOp>(
      loc, toTensor(outputType), inputs, concatAxisAttr);
}

Value OnnxBuilder::constant(Attribute denseAttr) const {
  return b.create<ONNXConstantOp>(loc, Attribute(), denseAttr);
}

Value OnnxBuilder::constantInt64(const ArrayRef<int64_t> intVals) const {
  Attribute denseAttr = b.getI64TensorAttr(intVals);
  return constant(denseAttr);
}

Value OnnxBuilder::constantFromRawBuffer(Type resultType, char *buf) const {
  DenseElementsAttr denseAttr =
      createDenseElementsAttrFromRawBuffer(resultType, buf);
  return b.create<ONNXConstantOp>(loc, resultType, Attribute(), denseAttr,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(),
      ArrayAttr());
}

Value OnnxBuilder::dim(Value input, int axis) const {
  Type resultType = RankedTensorType::get({1}, b.getI64Type());
  IntegerAttr axisAttr =
      IntegerAttr::get(b.getIntegerType(64, /*isSigned=*/true),
          APInt(64, axis, /*isSigned=*/true));
  return b.create<ONNXDimOp>(loc, resultType, input, axisAttr);
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

Value OnnxBuilder::mul(Type resultType, Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return b.create<ONNXMulOp>(loc, resultType, toTensor(A), toTensor(B));
}

Value OnnxBuilder::reduceSum(Type outputType, Value data, Value axes,
    bool keepDims, bool noop_with_empty_axes) const {
  int64_t i_keepDims = keepDims; // 0 if false, 1 if true
  int64_t i_noop_with_empty_axes = noop_with_empty_axes; // ditto
  return b.create<ONNXReduceSumOp>(loc, toTensor(outputType), toTensor(data),
      toTensor(axes), i_keepDims, i_noop_with_empty_axes);
}

Value OnnxBuilder::reshape(Type outputType, Value input, Value shape) const {
  return b.create<ONNXReshapeOp>(
      loc, toTensor(outputType), toTensor(input), toTensor(shape));
}

Value OnnxBuilder::shape(Type outputType, Value input) const {
  return b.create<ONNXShapeOp>(loc, toTensor(outputType), toTensor(input));
}

Value OnnxBuilder::slice(Type outputType, Value input, Value starts, Value ends,
    Value axes, Value steps) const {
  return b.create<ONNXSliceOp>(loc, toTensor(outputType), toTensor(input),
      toTensor(starts), toTensor(ends), toTensor(axes), toTensor(steps));
}

// 1D slice: take ints instead of values, and axis is by default 0 since we deal
// here with 1D vectors.
Value OnnxBuilder::slice(Type outputType, Value input, int64_t start,
    int64_t end, int64_t step) const {
  Value zeroVal = constant(b.getI64TensorAttr(ArrayRef<int64_t>({0})));
  Value startVal = constant(b.getI64TensorAttr(ArrayRef<int64_t>({start})));
  Value endVal = constant(b.getI64TensorAttr(ArrayRef<int64_t>({end})));
  Value stepVal = constant(b.getI64TensorAttr(ArrayRef<int64_t>({step})));
  return slice(outputType, input, startVal, endVal, /*axis*/ zeroVal, stepVal);
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

// =============================================================================
// More advanced operations
// =============================================================================

// Reshape input value "val" to a "N" dimensional vector. When
// "collapseMostSignificant" is true, then we collapse the R-N+1 most
// significant dimensions and keep the N-1 least significant dimensions as is.
//
// e.g. val has type 2x3x4x5xf32
// reshape(val, 3, true) -> reshape([-1, 4, 5], val, [-1, 4, 5])

// When  "collapseMostSignificant" is false, then we collapse the R-N+1 least
// significant dimensions and keep the N-1 most significant dimensions as is.
//
// e.g. reshape(val, 3, false) -> reshape([2, 3, -1], val, [2, 3, -1])

Value OnnxBuilder::reshapeToNDim(
    Value val, int64_t N, bool collapseMostSignificant) const {
  // Get rank of the original shape and determine if we have anything to do.
  int64_t rank = val.getType().cast<RankedTensorType>().getRank();
  int64_t keep = N - 1; // 1 dim for collapsed dims, keep N -1 from original.
  assert(rank >= N && "Require rank >= N");
  if (rank == N)
    // No collapse is needed, return self.
    return val;
  // Compute types.
  ArrayRef<int64_t> inputShape = val.getType().cast<ShapedType>().getShape();
  Type elementType = val.getType().cast<ShapedType>().getElementType();
  Type inputShapeType = RankedTensorType::get({rank}, b.getI64Type());
  Type keepShapeType = RankedTensorType::get({keep}, b.getI64Type());
  Type outputShapeType = RankedTensorType::get({N}, b.getI64Type());
  // Get input shape value.
  Value inputShapeVals = shape(inputShapeType, val);
  // Construct ONNX constants.
  Value minusOneVal = constantInt64({-1});
  // Shape values that we keep.
  int64_t start = collapseMostSignificant ? rank - keep : 0; // Inclusive.
  int64_t end = collapseMostSignificant ? rank : N - 1;      // Exclusive.
  Value keepVals =
      slice(keepShapeType, inputShapeVals, start, end, /*steps*/ 1);
  // Concat -1 and keep vals
  Value newShapeVals;
  if (collapseMostSignificant)
    // NewShapeVal is [-1,M,N] where M & N are the kept vals from the input.
    newShapeVals =
        concat(outputShapeType, ValueRange({minusOneVal, keepVals}), 0);
  else
    // NewShapeVal is [M,N,-1] where M & N are the kept vals from the input.
    newShapeVals =
        concat(outputShapeType, ValueRange({keepVals, minusOneVal}), 0);
  // Shape inference will infer the correct shape later, thus use -1 for
  // collapsed dims.
  llvm::SmallVector<int64_t, 4> outputDims;
  if (collapseMostSignificant)
    outputDims.emplace_back(-1);
  for (int i = start; i < end; ++i)
    outputDims.emplace_back(inputShape[i]);
  if (!collapseMostSignificant)
    outputDims.emplace_back(-1);
  Type outputType = RankedTensorType::get(outputDims, elementType);
  return reshape(outputType, val, newShapeVals);
}

} // namespace onnx_mlir
