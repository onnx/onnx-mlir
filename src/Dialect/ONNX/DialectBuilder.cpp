/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- DialectBuilder.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains builder functions for ONNX Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

// Identity affine
namespace onnx_mlir {

//====-------------------------- ONNX Builder ---------------------------===//

IntegerAttr OnnxBuilder::getSignedInt64Attr(int64_t n) const {
  return IntegerAttr::get(b().getIntegerType(64, /*isSigned=*/true), n);
}

// =============================================================================
// Basic operations
// =============================================================================

Value OnnxBuilder::add(Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return createOpAndInferShapes<ONNXAddOp>(toTensor(A), toTensor(B));
}

Value OnnxBuilder::cast(Value input, TypeAttr to) const {
  Type resultType;
  if (input.getType().cast<ShapedType>().hasRank())
    resultType = RankedTensorType::get(
        input.getType().cast<ShapedType>().getShape(), to.getValue());
  else
    resultType = UnrankedTensorType::get(to.getValue());
  IntegerAttr saturate = nullptr;
  return createTypedOpAndInferShapes<ONNXCastOp>(
      resultType, input, saturate, to);
}

Value OnnxBuilder::cast(Value input, Type to) const {
  return cast(input, TypeAttr::get(to));
}

Value OnnxBuilder::ceil(Value input) const {
  return createOpAndInferShapes<ONNXCeilOp>(toTensor(input.getType()), input);
}

Value OnnxBuilder::clip(
    Value input, Value min, Value max, bool scalarType) const {
  if (scalarType)
    return b().create<ONNXClipOp>(loc(), input.getType(), input, min, max);
  else
    return createOpAndInferShapes<ONNXClipOp>(toTensor(input.getType()),
        toTensor(input), toTensor(min), toTensor(max));
}

Value OnnxBuilder::concat(
    Type outputType, ValueRange inputs, int64_t axis) const {
  IntegerAttr concatAxisAttr = getSignedInt64Attr(axis);
  return createTypedOpAndInferShapes<ONNXConcatOp>(
      toTensor(outputType), inputs, concatAxisAttr);
}

Value OnnxBuilder::constant(Attribute denseAttr) const {
  assert((isa<DenseElementsAttr, DisposableElementsAttr>(denseAttr)) &&
         "unsupported onnx constant value attribute");
  return createOpAndInferShapes<ONNXConstantOp>(Attribute(), denseAttr);
}

Value OnnxBuilder::constantInt64(const ArrayRef<int64_t> intVals) const {
  Attribute denseAttr = b().getI64TensorAttr(intVals);
  return constant(denseAttr);
}

Value OnnxBuilder::conv(Type Y, Value X, Value W, Value B, StringRef autoPad,
    ArrayRef<int64_t> dilations, int64_t group, ArrayRef<int64_t> kernelShape,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> strides) const {
  StringAttr autoPadAttr = b().getStringAttr(autoPad);
  ArrayAttr dilationsAttr = b().getI64ArrayAttr(dilations);
  IntegerAttr groupAttr =
      IntegerAttr::get(b().getIntegerType(64, /*isSigned=*/true),
          APInt(64, group, /*isSigned=*/true));
  ArrayAttr kernelShapeAttr = b().getI64ArrayAttr(kernelShape);
  ArrayAttr padsAttr = b().getI64ArrayAttr(pads);
  ArrayAttr stridesAttr = b().getI64ArrayAttr(strides);
  return createOpAndInferShapes<ONNXConvOp>(toTensor(Y), X, W, B, autoPadAttr,
      dilationsAttr, groupAttr, kernelShapeAttr, padsAttr, stridesAttr);
}

Value OnnxBuilder::dim(Value input, int axis) const {
  Type resultType = RankedTensorType::get({1}, b().getI64Type());
  IntegerAttr axisAttr = getSignedInt64Attr(axis);
  return createTypedOpAndInferShapes<ONNXDimOp>(resultType, input, axisAttr);
}

void OnnxBuilder::dimGroup(Value input, int axis, int groupID) const {
  IntegerAttr axisAttr = getSignedInt64Attr(axis);
  IntegerAttr groupIDAttr = getSignedInt64Attr(groupID);
  // No shape needed for this one I believe.
  b().create<ONNXDimGroupOp>(loc(), input, axisAttr, groupIDAttr);
}

Value OnnxBuilder::div(Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return createOpAndInferShapes<ONNXDivOp>(toTensor(A), toTensor(B));
}

Value OnnxBuilder::expand(Type outputType, Value input, Value shape) const {
  return createOpAndInferShapes<ONNXExpandOp>(
      outputType, toTensor(input), toTensor(shape));
}

// ONNXLayerNormalizationOp, version with one output only (Y).
Value OnnxBuilder::layerNorm(Type outputType, Value input, Value scale,
    Value bias, int64_t axis, FloatAttr epsilon) const {
  IntegerAttr axisAttr = getSignedInt64Attr(axis);
  IntegerAttr stashTypeAttr = getSignedInt64Attr(1);
  Value noneVal = none();
  Type noneType = noneVal.getType();
  ONNXLayerNormalizationOp layerNormOp =
      createOpAndInferShapes<ONNXLayerNormalizationOp>(
          /*Y type*/ toTensor(outputType), /*mean type*/ noneType,
          /*std dev Type*/ noneType, toTensor(input), toTensor(scale),
          toTensor(bias), axisAttr, epsilon, stashTypeAttr);
  return layerNormOp.getY();
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
    return createOpAndInferShapes<ONNXGemmOp>(Y, aValue, bValue, none(),
        /*alpha=*/b().getF32FloatAttr(1.0), /*beta=*/b().getF32FloatAttr(1.0),
        /*transA=*/getSignedInt64Attr(0),
        /*transB=*/getSignedInt64Attr(0));
  return createOpAndInferShapes<ONNXMatMulOp>(toTensor(Y), aValue, bValue);
}

Value OnnxBuilder::max(ValueRange inputs) const {
  assert(inputs.size() >= 1 && "Expect at least one input");
  Type elementType = getElementType(inputs[0].getType());
  UnrankedTensorType outputType = UnrankedTensorType::get(elementType);
  return createTypedOpAndInferShapes<ONNXMaxOp>(outputType, inputs);
}

Value OnnxBuilder::min(ValueRange inputs) const {
  assert(inputs.size() >= 1 && "Expect at least one input");
  Type elementType = getElementType(inputs[0].getType());
  UnrankedTensorType outputType = UnrankedTensorType::get(elementType);
  return createTypedOpAndInferShapes<ONNXMinOp>(outputType, inputs);
}

Value OnnxBuilder::mul(Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return createOpAndInferShapes<ONNXMulOp>(toTensor(A), toTensor(B));
}

Value OnnxBuilder::mul(Type resultType, Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return createTypedOpAndInferShapes<ONNXMulOp>(
      resultType, toTensor(A), toTensor(B));
}

Value OnnxBuilder::none() const { return b().create<ONNXNoneOp>(loc()); }

Value OnnxBuilder::pad(
    Value input, Value pads, Value constantValue, std::string mode) const {
  Type elementType = getElementType(input.getType());
  Type outputType = UnrankedTensorType::get(elementType);
  Value constant = constantValue.getType().isa<NoneType>()
                       ? constantValue
                       : toTensor(constantValue);
  return createTypedOpAndInferShapes<ONNXPadOp>(toTensor(outputType),
      toTensor(input), toTensor(pads), constant,
      b().createOrFold<ONNXNoneOp>(loc()), b().getStringAttr(mode));
}

Value OnnxBuilder::padZero(Value input, Value pads) const {
  return pad(input, pads, b().create<ONNXNoneOp>(loc()), "constant");
}

Value OnnxBuilder::reduceMax(Type outputType, Value data, Value axes,
    bool keepDims, bool noop_with_empty_axes) const {
  int64_t i_keepDims = keepDims; // 0 if false, 1 if true
  int64_t i_noop_with_empty_axes = noop_with_empty_axes; // ditto
  return createTypedOpAndInferShapes<ONNXReduceMaxOp>(toTensor(outputType),
      toTensor(data), toTensor(axes), i_keepDims, i_noop_with_empty_axes);
}

Value OnnxBuilder::reduceMean(Type outputType, Value data, Value axes,
    bool keepDims, bool noop_with_empty_axes) const {
  int64_t i_keepDims = keepDims; // 0 if false, 1 if true
  int64_t i_noop_with_empty_axes = noop_with_empty_axes; // ditto
  return createTypedOpAndInferShapes<ONNXReduceMeanOp>(toTensor(outputType),
      toTensor(data), toTensor(axes), i_keepDims, i_noop_with_empty_axes);
}

Value OnnxBuilder::reduceMin(Type outputType, Value data, Value axes,
    bool keepDims, bool noop_with_empty_axes) const {
  int64_t i_keepDims = keepDims; // 0 if false, 1 if true
  int64_t i_noop_with_empty_axes = noop_with_empty_axes; // ditto
  return createTypedOpAndInferShapes<ONNXReduceMinOp>(toTensor(outputType),
      toTensor(data), toTensor(axes), i_keepDims, i_noop_with_empty_axes);
}

Value OnnxBuilder::reduceSum(Type outputType, Value data, Value axes,
    bool keepDims, bool noop_with_empty_axes) const {
  int64_t i_keepDims = keepDims; // 0 if false, 1 if true
  int64_t i_noop_with_empty_axes = noop_with_empty_axes; // ditto
  return createTypedOpAndInferShapes<ONNXReduceSumOp>(toTensor(outputType),
      toTensor(data), toTensor(axes), i_keepDims, i_noop_with_empty_axes);
}

Value OnnxBuilder::reciprocal(Value input) const {
  Type outputType = input.getType(); // input == output type.
  return createTypedOpAndInferShapes<ONNXReciprocalOp>(
      toTensor(outputType), toTensor(input));
}

Value OnnxBuilder::reshape(Type outputType, Value input, Value shape) const {
  return createTypedOpAndInferShapes<ONNXReshapeOp>(
      toTensor(outputType), toTensor(input), toTensor(shape));
}

Value OnnxBuilder::reshape(Type outputType, Value input, Value shape,
    mlir::IntegerAttr allowZero) const {
  return createTypedOpAndInferShapes<ONNXReshapeOp>(
      toTensor(outputType), toTensor(input), toTensor(shape), allowZero);
}

Value OnnxBuilder::reverseSequence(Type outputType, Value input,
    Value sequenceLens, int64_t batchAxis, int64_t timeAxis) const {
  IntegerAttr batchAxisAttr = getSignedInt64Attr(batchAxis);
  IntegerAttr timeAxisAttr = getSignedInt64Attr(timeAxis);
  return createTypedOpAndInferShapes<ONNXReverseSequenceOp>(
      toTensor(outputType), toTensor(input), toTensor(sequenceLens),
      batchAxisAttr, timeAxisAttr);
}

Value OnnxBuilder::round(Value input, bool scalarType) const {
  if (scalarType)
    return b().create<ONNXRoundOp>(loc(), input.getType(), input);
  else
    return createOpAndInferShapes<ONNXRoundOp>(
        toTensor(input.getType()), toTensor(input));
}

Value OnnxBuilder::shape(Type outputType, Value input) const {
  return createTypedOpAndInferShapes<ONNXShapeOp>(
      toTensor(outputType), toTensor(input));
}

Value OnnxBuilder::shape(Type outputType, Value input, int64_t start) const {
  IntegerAttr startAttr = getSignedInt64Attr(start);
  return createTypedOpAndInferShapes<ONNXShapeOp>(
      toTensor(outputType), toTensor(input), nullptr, startAttr);
}

Value OnnxBuilder::shape(
    Type outputType, Value input, int64_t start, int64_t end) const {
  IntegerAttr startAttr = getSignedInt64Attr(start);
  IntegerAttr endAttr = getSignedInt64Attr(end);
  return createTypedOpAndInferShapes<ONNXShapeOp>(
      toTensor(outputType), toTensor(input), endAttr, startAttr);
}

Value OnnxBuilder::slice(Type outputType, Value input, Value starts, Value ends,
    Value axes, Value steps) const {
  return createTypedOpAndInferShapes<ONNXSliceOp>(toTensor(outputType),
      toTensor(input), toTensor(starts), toTensor(ends), toTensor(axes),
      toTensor(steps));
}

// 1D slice: take ints instead of values, and axis is by default 0 since we deal
// here with 1D vectors.
Value OnnxBuilder::slice(Type outputType, Value input, int64_t start,
    int64_t end, int64_t step) const {
  Value zeroVal = constant(b().getI64TensorAttr(ArrayRef<int64_t>({0})));
  Value startVal = constant(b().getI64TensorAttr(ArrayRef<int64_t>({start})));
  Value endVal = constant(b().getI64TensorAttr(ArrayRef<int64_t>({end})));
  Value stepVal = constant(b().getI64TensorAttr(ArrayRef<int64_t>({step})));
  return slice(outputType, input, startVal, endVal, /*axis*/ zeroVal, stepVal);
}

Value OnnxBuilder::sqrt(Value input) const {
  return createOpAndInferShapes<ONNXSqrtOp>(toTensor(input));
}

ValueRange OnnxBuilder::split(
    TypeRange outputTypes, Value input, Value split, int64_t axis) const {
  IntegerAttr axisAttr = getSignedInt64Attr(axis);
  return createOpAndInferShapes<ONNXSplitOp>(toTensors(outputTypes),
      toTensor(input), toTensor(split), axisAttr, IntegerAttr())
      .getResults();
}

Value OnnxBuilder::squeeze(Type outputType, Value data, Value axes) const {
  return createTypedOpAndInferShapes<ONNXSqueezeOp>(
      toTensor(outputType), toTensor(data), toTensor(axes));
}

Value OnnxBuilder::sub(Value A, Value B) const {
  assert((A.getType().cast<ShapedType>().getElementType() ==
             B.getType().cast<ShapedType>().getElementType()) &&
         "A and B must have the same element type");
  return createOpAndInferShapes<ONNXSubOp>(toTensor(A), toTensor(B));
}

Value OnnxBuilder::sum(Type outputType, ValueRange inputs) const {
  return createTypedOpAndInferShapes<ONNXSumOp>(toTensor(outputType), inputs);
}

Value OnnxBuilder::transpose(
    Type outputType, Value input, ArrayAttr perm) const {
  return createTypedOpAndInferShapes<ONNXTransposeOp>(
      toTensor(outputType), toTensor(input), perm);
}

Value OnnxBuilder::transposeInt64(
    Value input, ArrayRef<int64_t> intPerm) const {
  Type elementType = getElementType(input.getType());
  Type outputType = UnrankedTensorType::get(elementType);
  return transpose(outputType, input, b().getI64ArrayAttr(intPerm));
}

Value OnnxBuilder::toTensor(Value input) const {
  // None input.
  if (isNoneValue(input))
    return input;
  if (input.getType().isa<TensorType>())
    return input;
  assert(input.getType().isa<MemRefType>() &&
         "expect RankedMemref type when not a TensorType");
  auto aTensorTy = toTensor(input.getType());
  // No shape inference for this op.
  return b()
      .create<UnrealizedConversionCastOp>(loc(), aTensorTy, input)
      .getResult(0);
}

TensorType OnnxBuilder::toTensor(Type input) const {
  if (auto tensorType = input.dyn_cast<TensorType>())
    return tensorType;
  assert(input.isa<MemRefType>() &&
         "expect RankedMemref type when not a TensorType");
  auto aTy = input.cast<ShapedType>();
  Type elementTy = aTy.getElementType();
  if (elementTy.isa<IndexType>()) {
    elementTy = b().getIntegerType(64);
  }
  return RankedTensorType::get(aTy.getShape(), elementTy);
}

TypeRange OnnxBuilder::toTensors(TypeRange inputs) const {
  assert(inputs.size() >= 2 && "Expect at least two inputs");
  if (llvm::all_of(inputs, [](Type t) { return (t.isa<TensorType>()); }))
    return inputs;
  assert(llvm::all_of(inputs, [](Type t) { return (t.isa<MemRefType>()); }) &&
         "All inputs expect RankedMemref type when not a TensorType");
  llvm::SmallVector<Type, 4> resultTypes;
  for (uint64_t i = 0; i < inputs.size(); ++i) {
    ShapedType aTy = inputs[i].cast<ShapedType>();
    Type elementTy = aTy.getElementType();
    if (elementTy.isa<IndexType>()) {
      elementTy = b().getIntegerType(64);
    }
    resultTypes.emplace_back(RankedTensorType::get(aTy.getShape(), elementTy));
  }
  return TypeRange(resultTypes);
}

Value OnnxBuilder::toMemref(Value input) const {
  if (input.getType().isa<MemRefType>())
    return input;
  assert(input.getType().isa<RankedTensorType>() &&
         "expect RankedMemref type when not a TensorType");
  auto aTy = input.getType().cast<ShapedType>();
  auto aTensorTy = MemRefType::get(aTy.getShape(), aTy.getElementType());
  // No shape inference for this op.
  return b()
      .create<UnrealizedConversionCastOp>(loc(), aTensorTy, input)
      .getResult(0);
}

Value OnnxBuilder::unsqueeze(Type outputType, Value data, Value axes) const {
  return createTypedOpAndInferShapes<ONNXUnsqueezeOp>(
      toTensor(outputType), toTensor(data), toTensor(axes));
}

Value OnnxBuilder::where(
    Type outputType, Value condition, Value X, Value Y) const {
  return createTypedOpAndInferShapes<ONNXWhereOp>(
      toTensor(outputType), toTensor(condition), toTensor(X), toTensor(Y));
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
  Type elementType = getElementType(val.getType());
  Type inputShapeType = RankedTensorType::get({rank}, b().getI64Type());
  Type keepShapeType = RankedTensorType::get({keep}, b().getI64Type());
  Type outputShapeType = RankedTensorType::get({N}, b().getI64Type());
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
    outputDims.emplace_back(ShapedType::kDynamic);
  for (int i = start; i < end; ++i)
    outputDims.emplace_back(inputShape[i]);
  if (!collapseMostSignificant)
    outputDims.emplace_back(ShapedType::kDynamic);
  Type outputType = RankedTensorType::get(outputDims, elementType);
  return reshape(outputType, val, newShapeVals);
}

// =============================================================================
// IndexExpr Builder for Analysis
// =============================================================================

// Return null if none is found.
ElementsAttr IndexExprBuilderForAnalysis::getConst(Value value) {
  return getElementAttributeFromONNXValue(value);
}

// Return null if the value at index i is not a constant.
Value IndexExprBuilderForAnalysis::getVal(Value intArrayVal, uint64_t i) {
  // Value, e.g. `tensor<3xi64>`, may come from `onnx.Concat` of constant and
  // runtime values. Thus, we potentially get a constant value at index i.
  if (areDimsFromConcat(intArrayVal)) {
    SmallVector<Value, 4> dims;
    getDims(intArrayVal, dims);
    if (isDenseONNXConstant(dims[i]))
      return dims[i];
  }
  // Otherwise, for analysis, never create values, so return null.
  return nullptr;
}

Value IndexExprBuilderForAnalysis::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  return nullptr;
}

} // namespace onnx_mlir
