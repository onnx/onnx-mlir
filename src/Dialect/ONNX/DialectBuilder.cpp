/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- DialectBuilder.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019-2024 The IBM Research Authors.
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

Value OnnxBuilder::abs(Value input) const {
  Type outputType = input.getType(); // input == output type.
  return createTypedOpAndInferShapes<ONNXAbsOp>(
      toTensor(outputType), toTensor(input));
}

Value OnnxBuilder::add(Value A, Value B) const {
  assert((mlir::cast<ShapedType>(A.getType()).getElementType() ==
             mlir::cast<ShapedType>(B.getType()).getElementType()) &&
         "A and B must have the same element type");
  return createOpAndInferShapes<ONNXAddOp>(toTensor(A), toTensor(B));
}

Value OnnxBuilder::cast(Type outputType, Value input, IntegerAttr saturate,
    TypeAttr to, bool inferShape) const {
  if (inferShape)
    return createTypedOpAndInferShapes<ONNXCastOp>(
        outputType, input, saturate, to);
  else
    return b().create<ONNXCastOp>(loc(), outputType, input, saturate, to);
}

Value OnnxBuilder::cast(Value input, IntegerAttr saturate, TypeAttr to) const {
  Type resultType;
  if (mlir::cast<ShapedType>(input.getType()).hasRank())
    resultType = RankedTensorType::get(
        mlir::cast<ShapedType>(input.getType()).getShape(), to.getValue());
  else
    resultType = UnrankedTensorType::get(to.getValue());
  return createTypedOpAndInferShapes<ONNXCastOp>(
      resultType, input, saturate, to);
}

Value OnnxBuilder::cast(Value input, TypeAttr to) const {
  Type resultType;
  if (mlir::cast<ShapedType>(input.getType()).hasRank())
    resultType = RankedTensorType::get(
        mlir::cast<ShapedType>(input.getType()).getShape(), to.getValue());
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

Value OnnxBuilder::dequantizeLinear(
    Type resType, Value X, Value scale, Value zeroPoint, int axis) const {
  IntegerAttr axisAttr = getSignedInt64Attr(axis);
  return createOpAndInferShapes<ONNXDequantizeLinearOp>(
      resType, toTensor(X), toTensor(scale), toTensor(zeroPoint), axisAttr);
}

Value OnnxBuilder::div(Value A, Value B) const {
  assert((mlir::cast<ShapedType>(A.getType()).getElementType() ==
             mlir::cast<ShapedType>(B.getType()).getElementType()) &&
         "A and B must have the same element type");
  return createOpAndInferShapes<ONNXDivOp>(toTensor(A), toTensor(B));
}

Value OnnxBuilder::expand(Type outputType, Value input, Value shape) const {
  return createOpAndInferShapes<ONNXExpandOp>(
      outputType, toTensor(input), toTensor(shape));
}

Value OnnxBuilder::gelu(Value input, StringAttr approximateAttr) const {
  return createOpAndInferShapes<ONNXGeluOp>(
      toTensor(input.getType()), input, approximateAttr);
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
// In the case of GroupNormalization when stashType can be specified
Value OnnxBuilder::layerNorm(Type outputType, Value input, Value scale,
    Value bias, int64_t axis, FloatAttr epsilon, IntegerAttr stashType) const {
  IntegerAttr axisAttr = getSignedInt64Attr(axis);
  Value noneVal = none();
  Type noneType = noneVal.getType();
  ONNXLayerNormalizationOp layerNormOp =
      createOpAndInferShapes<ONNXLayerNormalizationOp>(
          /*Y type*/ toTensor(outputType), /*mean type*/ noneType,
          /*std dev Type*/ noneType, toTensor(input), toTensor(scale),
          toTensor(bias), axisAttr, epsilon, stashType);
  return layerNormOp.getY();
}

Value OnnxBuilder::qlinearMatMul(Type outputType, Value a, Value aScale,
    Value aZeroPoint, Value b, Value bScale, Value bZeroPoint, Value yScale,
    Value yZeroPoint) const {
  return createOpAndInferShapes<ONNXQLinearMatMulOp>(toTensor(outputType),
      toTensor(a), toTensor(aScale), toTensor(aZeroPoint), toTensor(b),
      toTensor(bScale), toTensor(bZeroPoint), toTensor(yScale),
      toTensor(yZeroPoint));
}

Value OnnxBuilder::RMSLayerNorm(Type outputType, Value input, Value scale,
    Value bias, int64_t axis, FloatAttr epsilon) const {
  IntegerAttr axisAttr = getSignedInt64Attr(axis);
  IntegerAttr stashTypeAttr = getSignedInt64Attr(1);
  Value noneVal = none();
  Type noneType = noneVal.getType();
  ONNXRMSLayerNormalizationOp RMSLayerNormOp =
      createOpAndInferShapes<ONNXRMSLayerNormalizationOp>(
          /*Y type*/ toTensor(outputType), /*std dev Type*/ noneType,
          toTensor(input), toTensor(scale), toTensor(bias), axisAttr, epsilon,
          stashTypeAttr);
  return RMSLayerNormOp.getY();
}

Value OnnxBuilder::matmul(Type Y, Value A, Value B, bool useGemm) const {
  // Gemm only supports rank 2.
  bool canUseGemm = useGemm && mlir::isa<ShapedType>(A.getType()) &&
                    mlir::cast<ShapedType>(A.getType()).hasRank() &&
                    (mlir::cast<ShapedType>(A.getType()).getRank() == 2) &&
                    mlir::isa<ShapedType>(B.getType()) &&
                    mlir::cast<ShapedType>(B.getType()).hasRank() &&
                    (mlir::cast<ShapedType>(B.getType()).getRank() == 2);
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
  assert((mlir::cast<ShapedType>(A.getType()).getElementType() ==
             mlir::cast<ShapedType>(B.getType()).getElementType()) &&
         "A and B must have the same element type");
  return createOpAndInferShapes<ONNXMulOp>(toTensor(A), toTensor(B));
}

Value OnnxBuilder::mul(Type resultType, Value A, Value B) const {
  assert((mlir::cast<ShapedType>(A.getType()).getElementType() ==
             mlir::cast<ShapedType>(B.getType()).getElementType()) &&
         "A and B must have the same element type");
  return createTypedOpAndInferShapes<ONNXMulOp>(
      resultType, toTensor(A), toTensor(B));
}

Value OnnxBuilder::none() const { return b().create<ONNXNoneOp>(loc()); }

Value OnnxBuilder::pad(
    Value input, Value pads, Value constantValue, std::string mode) const {
  Type elementType = getElementType(input.getType());
  Type outputType = UnrankedTensorType::get(elementType);
  Value constant = mlir::isa<NoneType>(constantValue.getType())
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

Value OnnxBuilder::reshape(
    Type outputType, Value input, Value shape, IntegerAttr allowZero) const {
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

Value OnnxBuilder::shape(Value input) const {
  int64_t rank = getRank(input.getType());
  Type outputType = RankedTensorType::get({rank}, b().getI64Type());
  return createTypedOpAndInferShapes<ONNXShapeOp>(
      toTensor(outputType), toTensor(input));
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

// Get the shape of an input and perform a permutation on it. Perm values are
// in the range [0, rank(input)). Type is inferred. Operation get the dimensions
// using onnx.dim and use onnx.concat to place the right value at the right
// position.
Value OnnxBuilder::shape(Value input, mlir::ArrayRef<int64_t> perm) const {
  ShapedType inputType = mlir::cast<ShapedType>(input.getType());
  int64_t inputRank = inputType.getRank();
  auto inputShape = inputType.getShape();
  int64_t permRank = perm.size();
  bool isStatic = llvm::none_of(
      inputShape, [](int64_t d) { return ShapedType::isDynamic(d); });
  if (isStatic) {
    // Static, no need to create dims. Gather shapes into a constant array.
    llvm::SmallVector<int64_t, 4> permutedShapes;
    for (int64_t p = 0; p < permRank; ++p) {
      int64_t d = perm[p];
      assert(d >= 0 && d < inputRank &&
             "perm values expected in [0..rank(input))");
      permutedShapes.emplace_back(inputShape[d]);
    }
    return constantInt64(permutedShapes);
  }
  // Dynamic shape: create the dims as needed and gather values in a concat.
  llvm::SmallVector<Value, 4> permutedDims;
  for (int64_t p = 0; p < permRank; ++p) {
    int64_t d = perm[p];
    assert(
        d >= 0 && d < inputRank && "perm values expected in [0..rank(input))");
    permutedDims.emplace_back(dim(input, d));
  }
  Type outputType = RankedTensorType::get({permRank}, b().getI64Type());
  return concat(outputType, permutedDims, 0);
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
  assert((mlir::cast<ShapedType>(A.getType()).getElementType() ==
             mlir::cast<ShapedType>(B.getType()).getElementType()) &&
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
  if (mlir::isa<TensorType>(input.getType()))
    return input;
  assert(mlir::isa<MemRefType>(input.getType()) &&
         "expect RankedMemref type when not a TensorType");
  auto aTensorTy = toTensor(input.getType());
  // No shape inference for this op.
  return b()
      .create<UnrealizedConversionCastOp>(loc(), aTensorTy, input)
      .getResult(0);
}

TensorType OnnxBuilder::toTensor(Type input) const {
  if (auto tensorType = mlir::dyn_cast<TensorType>(input))
    return tensorType;
  assert(mlir::isa<MemRefType>(input) &&
         "expect RankedMemref type when not a TensorType");
  auto aTy = mlir::cast<ShapedType>(input);
  Type elementTy = aTy.getElementType();
  if (mlir::isa<IndexType>(elementTy)) {
    elementTy = b().getIntegerType(64);
  }
  return RankedTensorType::get(aTy.getShape(), elementTy);
}

TypeRange OnnxBuilder::toTensors(TypeRange inputs) const {
  if (llvm::all_of(inputs, [](Type t) { return (mlir::isa<TensorType>(t)); }))
    return inputs;
  assert(llvm::all_of(inputs, [](Type t) {
    return (mlir::isa<MemRefType>(t));
  }) && "All inputs expect RankedMemref type when not a TensorType");
  llvm::SmallVector<Type, 4> resultTypes;
  for (uint64_t i = 0; i < inputs.size(); ++i) {
    ShapedType aTy = mlir::cast<ShapedType>(inputs[i]);
    Type elementTy = aTy.getElementType();
    if (mlir::isa<IndexType>(elementTy)) {
      elementTy = b().getIntegerType(64);
    }
    resultTypes.emplace_back(RankedTensorType::get(aTy.getShape(), elementTy));
  }
  return TypeRange(resultTypes);
}

Value OnnxBuilder::toMemref(Value input) const {
  if (mlir::isa<MemRefType>(input.getType()))
    return input;
  assert(mlir::isa<RankedTensorType>(input.getType()) &&
         "expect RankedMemref type when not a TensorType");
  auto aTy = mlir::cast<ShapedType>(input.getType());
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
  int64_t rank = mlir::cast<RankedTensorType>(val.getType()).getRank();
  int64_t keep = N - 1; // 1 dim for collapsed dims, keep N -1 from original.
  assert(rank >= N && "Require rank >= N");
  if (rank == N)
    // No collapse is needed, return self.
    return val;
  // Compute types.
  ArrayRef<int64_t> inputShape =
      mlir::cast<ShapedType>(val.getType()).getShape();
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
// Fold and emit support.
// =============================================================================

/// Emit an ONNXSqueezeOp. If the input is constant, do const propagation,
/// and return a constant.
Value OnnxBuilder::foldOrEmitONNXSqueezeOp(ConversionPatternRewriter &rewriter,
    Location loc, Type resultType, Value input, int64_t axis,
    DenseElementsAttrGetter getDenseElementAttrFromConstValue) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  TensorType tensorType = create.onnx.toTensor(resultType);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    DenseElementsAttr squeezedElements = inputElements.reshape(tensorType);
    return create.onnx.constant(squeezedElements);
  } else {
    return rewriter
        .create<ONNXSqueezeOp>(loc, tensorType, create.onnx.toTensor(input),
            create.onnx.constantInt64({axis}))
        .getResult();
  }
}

/// Emit an ONNXSqueezeV11Op. If the input is constant, do const propagation,
/// and return a constant.
Value OnnxBuilder::foldOrEmitONNXSqueezeV11Op(
    ConversionPatternRewriter &rewriter, Location loc, Type resultType,
    Value input, int64_t axis,
    DenseElementsAttrGetter getDenseElementAttrFromConstValue) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  TensorType tensorType = create.onnx.toTensor(resultType);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    DenseElementsAttr squeezedElements = inputElements.reshape(tensorType);
    return create.onnx.constant(squeezedElements);
  } else {
    return rewriter
        .create<ONNXSqueezeV11Op>(loc, tensorType, create.onnx.toTensor(input),
            rewriter.getI64ArrayAttr(axis))
        .getResult();
  }
}

/// Emit an ONNXUnsqueezeOp. If the input is constant, do const
/// propagation, and return a constant.
Value OnnxBuilder::foldOrEmitONNXUnsqueezeOp(
    ConversionPatternRewriter &rewriter, Location loc, Type resultType,
    Value input, int64_t axis,
    DenseElementsAttrGetter getDenseElementAttrFromConstValue) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  TensorType tensorType = create.onnx.toTensor(resultType);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    DenseElementsAttr unsqueezedElements = inputElements.reshape(tensorType);
    return create.onnx.constant(unsqueezedElements);
  } else {
    return rewriter
        .create<ONNXUnsqueezeOp>(loc, tensorType, create.onnx.toTensor(input),
            create.onnx.constantInt64({axis}))
        .getResult();
  }
}

/// Emit an ONNXUnsqueezeV11Op. If the input is constant, do const
/// propagation, and return a constant.
Value OnnxBuilder::foldOrEmitONNXUnsqueezeV11Op(
    ConversionPatternRewriter &rewriter, Location loc, Type resultType,
    Value input, int64_t axis,
    DenseElementsAttrGetter getDenseElementAttrFromConstValue) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  TensorType tensorType = create.onnx.toTensor(resultType);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    DenseElementsAttr unsqueezedElements = inputElements.reshape(tensorType);
    return create.onnx.constant(unsqueezedElements);
  } else {
    return rewriter
        .create<ONNXUnsqueezeV11Op>(loc, tensorType,
            create.onnx.toTensor(input), rewriter.getI64ArrayAttr(axis))
        .getResult();
  }
}

/// Emit an ONNXSplitOp. If the input is constant, do const propagation, and
/// return constants.
/// Only support evenly splitting.
std::vector<Value> OnnxBuilder::foldOrEmitONNXSplitOp(
    ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<Type> resultTypes, Value input, int64_t axis,
    DenseElementsAttrGetter getDenseElementAttrFromConstValue) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  std::vector<Value> resVals;
  int outputNum = resultTypes.size();
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    auto inputShape = inputElements.getType().getShape();
    assert(outputNum == 0 || inputShape[axis] % outputNum == 0);
    int64_t sizeOfEachSplit = outputNum != 0 ? inputShape[axis] / outputNum : 0;
    SmallVector<int64_t, 4> sizes(outputNum, sizeOfEachSplit);

    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    std::vector<ElementsAttr> splits =
        elementsBuilder.split(inputElements, axis, sizes);
    for (ElementsAttr splitElements : splits) {
      // Avoid DisposableElementsAttr during conversion.
      DenseElementsAttr denseSplitElements =
          elementsBuilder.toDenseElementsAttr(splitElements);
      Value constVal = create.onnx.constant(denseSplitElements);
      resVals.emplace_back(constVal);
    }
  } else {
    SmallVector<Type, 4> convertedTypes;
    SmallVector<int64_t> splitSizesI64;
    for (auto t : resultTypes) {
      convertedTypes.emplace_back(create.onnx.toTensor(t));
      splitSizesI64.emplace_back(mlir::cast<ShapedType>(t).getShape()[axis]);
    }
    Value splitSizes = create.onnx.constantInt64(splitSizesI64);
    ONNXSplitOp split = rewriter.create<ONNXSplitOp>(loc, convertedTypes,
        create.onnx.toTensor(input), splitSizes,
        /*axis=*/axis, nullptr);
    for (int i = 0; i < outputNum; ++i)
      resVals.emplace_back(split.getOutputs()[i]);
  }
  return resVals;
}

/// Emit an ONNXSplitV11Op. If the input is constant, do const propagation, and
/// return constants.
/// Only support evenly splitting.
std::vector<Value> OnnxBuilder::foldOrEmitONNXSplitV11Op(
    ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<Type> resultTypes, Value input, int64_t axis,
    DenseElementsAttrGetter getDenseElementAttrFromConstValue) {

  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  std::vector<Value> resVals;
  int outputNum = resultTypes.size();

  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    auto inputShape = inputElements.getType().getShape();
    assert(outputNum == 0 || inputShape[axis] % outputNum == 0);
    int64_t sizeOfEachSplit = outputNum != 0 ? inputShape[axis] / outputNum : 0;
    SmallVector<int64_t, 4> sizes(outputNum, sizeOfEachSplit);

    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    std::vector<ElementsAttr> splits =
        elementsBuilder.split(inputElements, axis, sizes);
    for (ElementsAttr splitElements : splits) {
      // Avoid DisposableElementsAttr during conversion.
      DenseElementsAttr denseSplitElements =
          elementsBuilder.toDenseElementsAttr(splitElements);
      resVals.emplace_back(create.onnx.constant(denseSplitElements));
    }
  } else {
    SmallVector<Type, 4> convertedTypes;
    for (auto t : resultTypes) {
      convertedTypes.emplace_back(create.onnx.toTensor(t));
    }
    ONNXSplitV11Op split = rewriter.create<ONNXSplitV11Op>(loc, convertedTypes,
        create.onnx.toTensor(input),
        /*axis=*/axis, nullptr);
    for (int i = 0; i < outputNum; ++i)
      resVals.emplace_back(split.getOutputs()[i]);
  }
  return resVals;
}

/// Emit an ONNXTransposeOp. If the input is constant, do const propagation,
/// and return a constant.
Value OnnxBuilder::foldOrEmitONNXTransposeOp(
    ConversionPatternRewriter &rewriter, Location loc, Type resultType,
    Value input, ArrayAttr permAttr,
    DenseElementsAttrGetter getDenseElementAttrFromConstValue) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  if (DenseElementsAttr inputElements =
          getDenseElementAttrFromConstValue(input)) {
    SmallVector<uint64_t, 4> perm;
    for (auto permVal : permAttr.getValue())
      perm.emplace_back(mlir::cast<IntegerAttr>(permVal).getInt());

    OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
    ElementsAttr transposedElements =
        elementsBuilder.transpose(inputElements, perm);
    // Avoid DisposableElementsAttr during conversion.
    DenseElementsAttr denseTransposedElements =
        elementsBuilder.toDenseElementsAttr(transposedElements);
    return create.onnx.constant(denseTransposedElements);
  } else {
    return rewriter
        .create<ONNXTransposeOp>(loc, create.onnx.toTensor(resultType),
            create.onnx.toTensor(input), permAttr)
        .getResult();
  }
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
