/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ONNXOpsHelper.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

// Identity affine
using namespace mlir;

namespace onnx_mlir {

AffineMap getIdentityDimMap(Builder &builder) {
  return AffineMap::get(1, 0, {builder.getAffineDimExpr(0)});
}

// Pool/conv affine
// dim =
//   let numerator = (input + pad - (kernel - 1) * dilation - 1)
//   in let denominator = stride
//      in
//        if (ceilMode)
//          ceil(numerator / denominator) + 1
//        else
//          floor(numerator / denominator) + 1
AffineMap getConvDimMap(Builder &builder, bool ceilMode) {
  AffineExpr input = builder.getAffineDimExpr(0);
  AffineExpr kernel = builder.getAffineSymbolExpr(0);
  AffineExpr pad = builder.getAffineSymbolExpr(1);
  AffineExpr stride = builder.getAffineSymbolExpr(2);
  AffineExpr dilation = builder.getAffineSymbolExpr(3);

  AffineExpr dimExp;
  if (ceilMode)
    dimExp = (input + pad - (kernel - 1) * dilation - 1).ceilDiv(stride) + 1;
  else
    dimExp = (input + pad - (kernel - 1) * dilation - 1).floorDiv(stride) + 1;

  return AffineMap::get(1, 4, {dimExp});
}

/// IndexExprs to compute the start and end indices of the convolution/pooling
/// window.
///
/// The conv/pooling window can be smaller than the kernel when slicing it over
/// the border edges. Thus, we will compute the start and end indices for
/// each window dimension as follows.
///   firstValidH = ceil(float(ptH / dH)) * dH - ptH
///   startH = max(firstValidH, ho * sH - ptH)
///   endH = min(H, ho * sH + (kH - 1) * dH  + 1 - pbH)
///
/// Full conv/pooling window can be reconstructed by:
///   hDim = round(float(endH - startH) / float(dH))
//
/// We also want to compute how the window is smaller than the kernel.
///   kernelOffset = min(0, ho * sH - ptH)
///
/// How to derive 'firstValidH':
///   When dilation is non-unit, the first valid pixel to apply conv/pooling on
///   will not be the 0-th pixel, but rather the smallest integer n to make
///   '-pH + n * dH' greater than or equal to 0, where pH and dH are pad
///   and dilation along axis H. We derive what is this smallest n:
///   -pH + n * dH >= 0
///         n * dH >= pH
///              n >= pH/dH
///   thus n = ceil(pH/dH)
///   thus the first valid pixel location is 'ceil(pH / dH) * dH- pH'.
///
/// This function returns {startH, endH, kernelOffset}.

std::vector<IndexExpr> getIndexExprsForConvWindow(
    SmallVectorImpl<IndexExpr> &inputExprs, bool ceilMode, bool isDilated) {
  assert(inputExprs.size() == 6 && "Not enough inputs");
  IndexExpr windowStartExpr, windowEndExpr, kernelOffsetExpr;
  IndexExpr outputIndex = inputExprs[0];
  IndexExpr inputDim = inputExprs[1];
  IndexExpr kernelDim = inputExprs[2];
  IndexExpr padTopDim = inputExprs[3];
  IndexExpr strideDim = inputExprs[4];
  IndexExpr dilationDim = inputExprs[5];

  IndexExpr start1 = (padTopDim).ceilDiv(dilationDim) * dilationDim - padTopDim;
  IndexExpr start2 = outputIndex * strideDim - padTopDim;
  IndexExpr end1 = inputDim;
  IndexExpr end2 =
      outputIndex * strideDim + (kernelDim - 1) * dilationDim + 1 - padTopDim;

  // windowStartExpr
  SmallVector<IndexExpr, 2> startExprs = {start1, start2};
  windowStartExpr = IndexExpr::max(startExprs);
  // windowEndExpr
  SmallVector<IndexExpr, 2> endExprs = {end1, end2};
  windowEndExpr = IndexExpr::min(endExprs);
  // kernelOffsetExpr
  SmallVector<IndexExpr, 2> kernelExprs = {LiteralIndexExpr(0), start2};
  kernelOffsetExpr = IndexExpr::min(kernelExprs);

  return std::vector<IndexExpr>{
      windowStartExpr, windowEndExpr, kernelOffsetExpr};
}

/// The conv/pooling window can be smaller than the kernel when slicing it over
/// the border edges. This function returns an AffineMap to compute the size of
/// one edge of the window.
AffineMap getWindowAffineMap(Builder &builder, bool ceilMode, bool isDilated) {
  AffineMap windowDimMap;
  // Compute start and end indices.
  AffineExpr outputIndex = builder.getAffineDimExpr(0);
  AffineExpr inputDim = builder.getAffineSymbolExpr(0);
  AffineExpr kernelDim = builder.getAffineSymbolExpr(1);
  AffineExpr padTopDim = builder.getAffineSymbolExpr(2);
  AffineExpr strideDim = builder.getAffineSymbolExpr(3);
  AffineExpr dilationDim = builder.getAffineSymbolExpr(4);
  AffineExpr start1 =
      (padTopDim).ceilDiv(dilationDim) * dilationDim - padTopDim;
  AffineExpr start2 = outputIndex * strideDim - padTopDim;
  AffineExpr end1 = inputDim;
  AffineExpr end2 =
      outputIndex * strideDim + (kernelDim - 1) * dilationDim + 1 - padTopDim;

  // Compute the window's size.
  SmallVector<AffineExpr, 4> dimExpr;
  // Upperbound for an affine.for is `min AffineMap`, where `min` is
  // automatically inserted when an affine.for is constructed from
  // an AffineMap, thus we rewrite `endH - startH` as follows:
  //   endH - startH
  //     = min(end1, end2) - max(start1, start2)
  //     = min(end1 - start1, end1 - start2, end2 - start1, end2 - start2)
  AffineExpr dimExpr1 = end1 - start1;
  AffineExpr dimExpr2 = end1 - start2;
  AffineExpr dimExpr3 = end2 - start1;
  AffineExpr dimExpr4 = end2 - start2;
  for (AffineExpr de : {dimExpr1, dimExpr2, dimExpr3, dimExpr4}) {
    if (isDilated) {
      de = de + 1;
      de = (ceilMode) ? de.ceilDiv(dilationDim) : de.floorDiv(dilationDim);
    }
    dimExpr.emplace_back(de);
  }
  windowDimMap = AffineMap::get(1, 5, dimExpr, builder.getContext());

  return windowDimMap;
}

//===----------------------------------------------------------------------===//
// ONNX Helper functions
//===----------------------------------------------------------------------===//

size_t ArrayAttrSize(ArrayAttr a) { return a.size(); }

size_t ArrayAttrSize(Optional<ArrayAttr> a) { return a.getValue().size(); }

int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
  return (a.getValue()[i]).cast<IntegerAttr>().getInt();
}

int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i) {
  return (a.getValue().getValue()[i]).cast<IntegerAttr>().getInt();
}

DenseElementsAttr getDenseElementAttributeFromONNXValue(Value value) {
  ONNXConstantOp constantOp = getONNXConstantOp(value);
  if (constantOp)
    return constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
  return nullptr;
}

// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value) {
  return dyn_cast_or_null<ONNXConstantOp>(value.getDefiningOp());
}

Value createONNXConstantOpWithDenseAttr(
    OpBuilder &builder, Location loc, Attribute dense) {
  return builder.create<ONNXConstantOp>(loc, Attribute(), dense);
}

// Use 0xi64 to represent a None for an optional integer input
Value createNoneIntegerConstant(PatternRewriter &rewriter, Location loc) {
  SmallVector<int64_t, 1> dims(1, 0);
  SmallVector<int64_t> values;
  auto tensorType =
      mlir::RankedTensorType::get(dims, rewriter.getIntegerType(64));
  auto denseAttr =
      mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
  return rewriter.create<ONNXConstantOp>(loc, Attribute(), denseAttr);
}

// Use 0xf32 to represent a None for an optional float input
Value createNoneFloatConstant(PatternRewriter &rewriter, Location loc) {
  SmallVector<int64_t, 1> dims(1, 0);
  SmallVector<float> values;
  auto tensorType = mlir::RankedTensorType::get(dims, rewriter.getF32Type());
  auto denseAttr =
      mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
  return rewriter.create<ONNXConstantOp>(loc, Attribute(), denseAttr);
}

// Returns true if the Value is defined by a unit constant.
bool isFromNone(Value v) {
  if (v.getDefiningOp() && dyn_cast_or_null<ONNXNoneOp>(v.getDefiningOp()))
    return true;

  if (v.getDefiningOp() &&
      dyn_cast_or_null<ONNXConstantOp>(v.getDefiningOp())) {
    auto c = dyn_cast<ONNXConstantOp>(v.getDefiningOp());
    if (c.value().hasValue() && c.valueAttr().isa<DenseElementsAttr>()) {
      auto d = c.valueAttr().cast<DenseElementsAttr>();
      auto shape = d.getType().dyn_cast<RankedTensorType>().getShape();
      if (shape.size() == 1 && shape[0] == 0)
        return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Get a broadcasted type for RankedTensorType and MemRefType.
//===----------------------------------------------------------------------===//
Type getBroadcastedRankedType(Type type1, Type type2, Type elementType) {
  if (type1.isa<RankedTensorType>() && type2.isa<RankedTensorType>())
    return OpTrait::util::getBroadcastedType(type1, type2, elementType);
  if (type1.isa<MemRefType>() && type2.isa<MemRefType>()) {
    // Construct RankedTensorType(s).
    if (!elementType)
      elementType = type1.cast<MemRefType>().getElementType();
    RankedTensorType ty1 =
        RankedTensorType::get(type1.cast<MemRefType>().getShape(), elementType);
    RankedTensorType ty2 =
        RankedTensorType::get(type2.cast<MemRefType>().getShape(), elementType);
    // Compute a broadcasted type.
    Type outputType = OpTrait::util::getBroadcastedType(ty1, ty2);
    // Construct a MemRefType.
    return MemRefType::get(
        outputType.cast<RankedTensorType>().getShape(), elementType);
  } else
    return {};
}

//===----------------------------------------------------------------------===//
// Support for transpose patterns.
//===----------------------------------------------------------------------===//

/// Compute the combined permute pattern from a pair of permute patterns.
ArrayAttr CombinedTransposePattern(PatternRewriter &rewriter,
    ArrayAttr firstPermAttr, ArrayAttr secondPermAttr) {
  // Read first permute vectors.
  SmallVector<int64_t, 4> initialPerm;
  for (auto firstPermVal : firstPermAttr.getValue())
    initialPerm.emplace_back(firstPermVal.cast<IntegerAttr>().getInt());
  // Read second permute vector. Use it as an index in the first permute
  // vector.
  SmallVector<int64_t, 4> resPerm;
  for (auto secondPermVal : secondPermAttr.getValue()) {
    auto index = secondPermVal.cast<IntegerAttr>().getInt();
    resPerm.emplace_back(initialPerm[index]);
  }
  // Convert to Array of Attributes.
  ArrayRef<int64_t> resPermRefs(resPerm);
  return rewriter.getI64ArrayAttr(resPermRefs);
}

/// Test if the permute pattern correspond to an identity pattern.
/// Identity patterns are {0, 1, 2, ... , rank -1}.
bool IsIdentityPermuteVector(ArrayAttr permAttr) {
  int64_t currentIndex = 0;
  for (auto permVal : permAttr.getValue())
    if (permVal.cast<IntegerAttr>().getInt() != currentIndex++)
      return false;
  return true;
}

/// Test if the value has the specified constant shape
bool HasSpecifiedConstantShape(Value value, Value shape) {
  if (!hasShapeAndRank(value) || !hasShapeAndRank(shape))
    return false;

  ArrayRef<int64_t> valueShape = value.getType().cast<ShapedType>().getShape();
  DenseElementsAttr shapeAttr = getDenseElementAttributeFromONNXValue(shape);
  if (shapeAttr == nullptr)
    return false;

  int64_t dimensionsOfShape = shapeAttr.getType().getShape()[0];
  if ((int64_t)valueShape.size() != dimensionsOfShape)
    return false;

  auto valueIt = shapeAttr.getValues<APInt>().begin();
  for (int64_t i = 0; i < dimensionsOfShape; i++) {
    int64_t value = (*valueIt++).getSExtValue();
    if (valueShape[i] != value)
      return false;
  }
  return true;
}

/// Test if two axis arrays contain the same values or not.
bool AreTheSameAxisArray(int64_t rank, ArrayAttr lhsAttr, ArrayAttr rhsAttr) {
  // false if one of the array attributes is null.
  if (!(lhsAttr) || !(rhsAttr))
    return false;

  SmallVector<int64_t, 4> lhs;
  for (auto attr : lhsAttr.getValue()) {
    int64_t axis = attr.cast<IntegerAttr>().getInt();
    if (axis < 0)
      axis += rank;
    lhs.emplace_back(axis);
  }

  size_t rhsSize = 0;
  for (auto attr : rhsAttr.getValue()) {
    int64_t axis = attr.cast<IntegerAttr>().getInt();
    if (axis < 0)
      axis += rank;
    // false if axis is not in the lhs. Early stop.
    if (!llvm::any_of(lhs, [&](int64_t lhsAxis) { return lhsAxis == axis; }))
      return false;
    rhsSize++;
  }

  // false if having different number of elements.
  if (lhs.size() != rhsSize)
    return false;

  return true;
}

/// Convert ConstantOp to ArrayAttr and test if they have the same values
bool AreTheSameConstantOpDenseAttr(
    Builder &builder, int64_t rank, Value lhsOp, Value rhsOp) {
  ONNXConstantOp lhsConstOp = dyn_cast<ONNXConstantOp>(lhsOp.getDefiningOp());
  ONNXConstantOp rhsConstOp = dyn_cast<ONNXConstantOp>(rhsOp.getDefiningOp());
  if (lhsConstOp && rhsConstOp) {
    auto lhsArrAttr = createArrayAttrFromConstantOp(builder, lhsConstOp);
    auto rhsArrAttr = createArrayAttrFromConstantOp(builder, rhsConstOp);
    return AreTheSameAxisArray(rank, lhsArrAttr, rhsArrAttr);
  } else {
    return false;
  }
}

/// Test if 'val' has shape and rank or not.
bool hasShapeAndRank(Value val) {
  return val.getType().isa<ShapedType>() &&
         val.getType().cast<ShapedType>().hasRank();
}

//===----------------------------------------------------------------------===//
// Support for rewrite patterns.
//===----------------------------------------------------------------------===//

// Create an ArrayAttr from a dense ConstantOp
ArrayAttr createArrayAttrFromConstantOp(Builder &builder, Value constOp) {
  auto denseAttr = getDenseElementAttributeFromONNXValue(constOp);
  assert(denseAttr && "ConstantOp is not a DenseElementsAttr");
  SmallVector<int64_t, 4> intVals;
  for (auto val : denseAttr.getValues<IntegerAttr>()) {
    intVals.emplace_back(val.getInt());
  }
  return builder.getI64ArrayAttr(ArrayRef<int64_t>(intVals));
}

// Create a DenseElementsAttr from a float attribute.
DenseElementsAttr createDenseElementsAttrFromFloatAttr(
    PatternRewriter &rewriter, Type elementType, FloatAttr attr) {
  SmallVector<int64_t, 1> dims(1, 1);
  SmallVector<float, 1> values(1, attr.getValue().convertToFloat());
  auto tensorType = RankedTensorType::get(dims, elementType);
  return DenseElementsAttr::get(tensorType, makeArrayRef(values));
}

// Create a DenseElementsAttr from a integer attribute.
// The attribute is assumed to be SingedInteger
DenseElementsAttr createDenseElementsAttrFromIntegerAttr(
    PatternRewriter &rewriter, Type elementType, IntegerAttr attr) {
  SmallVector<int64_t, 1> dims(1, 1);
  SmallVector<int64_t, 1> values(1, attr.getSInt());
  auto tensorType = RankedTensorType::get(dims, elementType);
  return DenseElementsAttr::get(tensorType, makeArrayRef(values));
}

DenseElementsAttr createDenseElementsAttrFromFloatAttrs(
    PatternRewriter &rewriter, Type elementType, SmallVector<Attribute> attrs) {
  SmallVector<int64_t, 1> dims(1, attrs.size());
  SmallVector<float, 1> values;
  for (auto attr : attrs) {
    values.push_back(attr.cast<FloatAttr>().getValue().convertToFloat());
  }
  auto tensorType = RankedTensorType::get(dims, elementType);
  return DenseElementsAttr::get(tensorType, makeArrayRef(values));
}

// Integer attribute is assumed to be Signedless
DenseElementsAttr createDenseElementsAttrFromIntegerAttrs(
    PatternRewriter &rewriter, Type elementType, SmallVector<Attribute> attrs) {
  SmallVector<int64_t, 1> dims(1, attrs.size());
  SmallVector<int64_t, 1> values;
  for (auto attr : attrs) {
    values.push_back(attr.cast<IntegerAttr>().getInt());
  }
  auto tensorType = RankedTensorType::get(dims, elementType);
  return DenseElementsAttr::get(tensorType, makeArrayRef(values));
}

// Create a DenseElementsAttr from a String attribute.
DenseElementsAttr createDenseElementsAttrFromStringAttrs(
    PatternRewriter &rewriter, Type elementType, SmallVector<Attribute> attrs) {
  SmallVector<int64_t, 1> dims(1, attrs.size());
  SmallVector<StringRef, 1> values;
  for (auto attr : attrs) {
    values.push_back(attr.cast<StringAttr>().getValue());
  }
  auto tensorType = RankedTensorType::get(dims, elementType);
  return DenseElementsAttr::get(tensorType, makeArrayRef(values));
}

Value normalizeConstantOp(
    PatternRewriter &rewriter, Value output, Attribute attr) {
  Type elementType;
  Type outputType = output.getType();
  if (outputType.dyn_cast<ShapedType>()) {
    elementType = outputType.cast<ShapedType>().getElementType();
  } else {
    elementType = outputType;
  }

  DenseElementsAttr denseAttr;
  if (attr.dyn_cast<FloatAttr>()) {
    denseAttr =
        createDenseElementsAttrFromFloatAttrs(rewriter, elementType, {attr});
  } else if (attr.dyn_cast<IntegerAttr>()) {
    denseAttr = createDenseElementsAttrFromIntegerAttr(
        rewriter, elementType, attr.cast<IntegerAttr>());
  } else if (attr.dyn_cast<StringAttr>()) {
    denseAttr =
        createDenseElementsAttrFromStringAttrs(rewriter, elementType, {attr});
  } else if (attr.dyn_cast<ArrayAttr>()) {
    ArrayAttr myAttr = attr.cast<ArrayAttr>();
    SmallVector<Attribute> attrs(
        myAttr.getValue().begin(), myAttr.getValue().end());
    if (attrs[0].dyn_cast<FloatAttr>()) {
      denseAttr =
          createDenseElementsAttrFromFloatAttrs(rewriter, elementType, attrs);
    } else if (attrs[0].dyn_cast<IntegerAttr>()) {
      denseAttr =
          createDenseElementsAttrFromIntegerAttrs(rewriter, elementType, attrs);
    } else if (attrs[0].dyn_cast<StringAttr>()) {
      denseAttr =
          createDenseElementsAttrFromStringAttrs(rewriter, elementType, attrs);
    } else {
      llvm_unreachable("unexpected Attribute");
    }
  } else {
    llvm_unreachable("unexpected Attribute");
  }
  return rewriter.create<ONNXConstantOp>(output.getLoc(), output.getType(),
      Attribute(), denseAttr, FloatAttr(), ArrayAttr(), IntegerAttr(),
      ArrayAttr(), StringAttr(), ArrayAttr());
}

// Create a DenseElementsAttr based on the shape of type.
DenseElementsAttr createDenseElementsAttrFromShape(
    PatternRewriter &rewriter, Value value) {
  auto inType = value.getType().cast<ShapedType>();
  auto shape = inType.getShape();
  SmallVector<int64_t, 1> dims = {inType.getRank()};
  SmallVector<int64_t, 4> values(shape.begin(), shape.end());
  auto tensorType = RankedTensorType::get(dims, rewriter.getIntegerType(64));
  return DenseElementsAttr::get(tensorType, makeArrayRef(values));
}

// Create a DenseElementsAttr based on the size of type.
DenseElementsAttr createDenseElementsAttrFromSize(
    PatternRewriter &rewriter, Value value) {
  auto inType = value.getType().cast<ShapedType>();
  SmallVector<int64_t, 1> dims(1, 1);
  SmallVector<int64_t, 1> values = {inType.getNumElements()};
  auto tensorType = RankedTensorType::get(dims, rewriter.getIntegerType(64));
  return DenseElementsAttr::get(tensorType, makeArrayRef(values));
}

/// Check whether a value is produced by a dense ONNXConstantOp.
bool isDenseONNXConstant(Value result) {
  Operation *op = result.getDefiningOp();

  ONNXConstantOp constOp = dyn_cast_or_null<ONNXConstantOp>(op);
  // Not a constant.
  if (!constOp)
    return false;

  // If the dense attribute is null, there must be buffer_id
  // attribute.
  if (!(op->getAttrOfType<Attribute>("value")))
    return false;
  // The other attributes must be null.
  if (op->getAttrOfType<Attribute>("sparse_value"))
    return false;
  if (op->getAttrOfType<Attribute>("value_float"))
    return false;
  if (op->getAttrOfType<Attribute>("value_floats"))
    return false;
  if (op->getAttrOfType<Attribute>("value_int"))
    return false;
  if (op->getAttrOfType<Attribute>("value_ints"))
    return false;
  if (op->getAttrOfType<Attribute>("value_string"))
    return false;
  if (op->getAttrOfType<Attribute>("value_strings"))
    return false;

  return true;
}

/// Get scalar value when it is a constant.
template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(DenseElementsAttr &denseAttr, Type type) {
  Type elementaryType = getElementTypeOrSelf(type);
  if (elementaryType.isInteger(16) || elementaryType.isInteger(32) ||
      elementaryType.isInteger(64)) {
    auto valueIt = denseAttr.getValues<IntegerAttr>().begin();
    return (RESULT_TYPE)(*valueIt).cast<IntegerAttr>().getInt();
  } else if (elementaryType.isF32()) {
    auto valueIt = denseAttr.getValues<APFloat>().begin();
    return (RESULT_TYPE)(*valueIt).convertToFloat();
  } else if (elementaryType.isF64()) {
    auto valueIt = denseAttr.getValues<APFloat>().begin();
    return (RESULT_TYPE)(*valueIt).convertToDouble();
  }
  llvm_unreachable("Unexpected type.");
  return 0;
}

template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(ONNXConstantOp constantOp, Type type) {
  DenseElementsAttr attr = constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
  if (!attr)
    constantOp.emitError("DenseElementsAttr expected");
  return getScalarValue<RESULT_TYPE>(attr, type);
}

// Template instantiation for getScalarValue

template double getScalarValue<double>(ONNXConstantOp constantOp, Type type);
template int64_t getScalarValue<int64_t>(ONNXConstantOp constantOp, Type type);

// Convert type to MLIR type.
// A complete list of types can be found in:
// <onnx-mlir-build-folder>/third_party/onnx/onnx/onnx.pb.h
// TODO: Update Int*/Uint* to emit signed/unsigned MLIR types
mlir::Type convertONNXTypeToMLIRType(
    mlir::OpBuilder &builder_, onnx::TensorProto_DataType onnxType) {
  switch (onnxType) {
  case onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16:
    return builder_.getBF16Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    return builder_.getF16Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
    return builder_.getF32Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
    return builder_.getF64Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
    return builder_.getIntegerType(/*width=*/8);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
    return builder_.getIntegerType(/*width=*/8, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
    return builder_.getIntegerType(/*width=*/16);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
    return builder_.getIntegerType(/*width=*/16, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
    return builder_.getIntegerType(/*width=*/32);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
    return builder_.getIntegerType(/*width=*/32, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
    return builder_.getIntegerType(/*width=*/64);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
    return builder_.getIntegerType(/*width=*/64, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
    return builder_.getI1Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
    return mlir::ONNXStringType::get(builder_.getContext());

  case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX64:
  case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX128:
  case onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED:
    llvm_unreachable("Unsupported data type encountered.");
    return nullptr;
  }

  llvm_unreachable("Unsupported data type encountered.");
}

} // namespace onnx_mlir
