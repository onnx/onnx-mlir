/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ONNXOpsHelper.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

// Identity affine
using namespace mlir;
using namespace mlir::onnxmlir;

//====-------------------------- ONNX Builder ---------------------------===//

Value OnnxBuilder::add(Value A, Value B) {
  return b.create<ONNXAddOp>(loc, A, B);
}

Value OnnxBuilder::sub(Value A, Value B) {
  return b.create<ONNXSubOp>(loc, A, B);
}

Value OnnxBuilder::mul(Value A, Value B) {
  return b.create<ONNXMulOp>(loc, A, B);
}

Value OnnxBuilder::div(Value A, Value B) {
  return b.create<ONNXDivOp>(loc, A, B);
}

Value OnnxBuilder::matmul(Type Y, Value A, Value B) {
  return b.create<ONNXMatMulOp>(loc, Y, A, B);
}

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
  SmallVector<mlir::IndexExpr, 2> startExprs = {start1, start2};
  windowStartExpr = IndexExpr::max(startExprs);
  // windowEndExpr
  SmallVector<mlir::IndexExpr, 2> endExprs = {end1, end2};
  windowEndExpr = IndexExpr::min(endExprs);
  // kernelOffsetExpr
  SmallVector<mlir::IndexExpr, 2> kernelExprs = {LiteralIndexExpr(0), start2};
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
  auto definingOp = value.getDefiningOp();
  if (auto constantOp = dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp)) {
    return constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
  }
  return nullptr;
}

// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value) {
  return dyn_cast_or_null<mlir::ONNXConstantOp>(value.getDefiningOp());
}

Value getONNXConstantOpFromDenseAttr(
    PatternRewriter &rewriter, Location loc, Attribute dense) {
  return rewriter.create<ONNXConstantOp>(loc, Attribute(), dense);
}

// Returns true if the Value is defined by none constant
bool isFromNone(Value v) {
  if (v.getDefiningOp() &&
      llvm::dyn_cast_or_null<mlir::ConstantOp>(v.getDefiningOp())) {
    mlir::ConstantOp c = llvm::dyn_cast<mlir::ConstantOp>(v.getDefiningOp());
    if (c.getValue().isa<UnitAttr>())
      return true;
  }
  if (v.getDefiningOp() &&
      llvm::dyn_cast_or_null<mlir::ONNXConstantOp>(v.getDefiningOp())) {
    mlir::ONNXConstantOp c =
        llvm::dyn_cast<mlir::ONNXConstantOp>(v.getDefiningOp());
    if (c.value().hasValue() && c.valueAttr().isa<DenseElementsAttr>()) {
      DenseElementsAttr d = c.valueAttr().cast<DenseElementsAttr>();
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
Type getBroadcastedRankedType(Type type1, Type type2) {
  if (type1.isa<RankedTensorType>() && type2.isa<RankedTensorType>())
    return OpTrait::util::getBroadcastedType(type1, type2);
  if (type1.isa<MemRefType>() && type2.isa<MemRefType>()) {
    // Construct RankedTensorType(s).
    Type elementType = type1.cast<MemRefType>().getElementType();
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
bool HasSpecifiedConstantShape(mlir::Value value, mlir::Value shape) {
  if (!value.getType().isa<ShapedType>()) {
    return false;
  }
  ArrayRef<int64_t> valueShape = value.getType().cast<ShapedType>().getShape();
  DenseElementsAttr shapeAttr = getDenseElementAttributeFromONNXValue(shape);
  if (shapeAttr == nullptr) {
    return false;
  }
  int64_t dimensionsOfShape = shapeAttr.getType().getShape()[0];
  if ((int64_t)valueShape.size() != dimensionsOfShape) {
    return false;
  }
  auto valueIt = shapeAttr.getIntValues().begin();
  for (int64_t i = 0; i < dimensionsOfShape; i++) {
    int64_t value = (*valueIt++).getSExtValue();
    if (valueShape[i] != value) {
      return false;
    }
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
  auto tensorType = mlir::RankedTensorType::get(dims, elementType);
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
}

// Create a DenseElementsAttr from a integer attribute.
// The attribute is assumed to be SingedInteger
DenseElementsAttr createDenseElementsAttrFromIntegerAttr(
    PatternRewriter &rewriter, Type elementType, IntegerAttr attr) {
  SmallVector<int64_t, 1> dims(1, 1);
  SmallVector<int64_t, 1> values(1, attr.getSInt());
  auto tensorType = mlir::RankedTensorType::get(dims, elementType);
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
}

DenseElementsAttr createDenseElementsAttrFromFloatAttrs(
    PatternRewriter &rewriter, Type elementType, SmallVector<Attribute> attrs) {
  SmallVector<int64_t, 1> dims(1, attrs.size());
  SmallVector<float, 1> values;
  for (auto attr : attrs) {
    values.push_back(attr.cast<FloatAttr>().getValue().convertToFloat());
  }
  auto tensorType = mlir::RankedTensorType::get(dims, elementType);
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
}

// Integer attribute is assumed to be Signedless
DenseElementsAttr createDenseElementsAttrFromIntegerAttrs(
    PatternRewriter &rewriter, Type elementType, SmallVector<Attribute> attrs) {
  SmallVector<int64_t, 1> dims(1, attrs.size());
  SmallVector<int64_t, 1> values;
  for (auto attr : attrs) {
    values.push_back(attr.cast<IntegerAttr>().getInt());
  }
  auto tensorType = mlir::RankedTensorType::get(dims, elementType);
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
}

// Create a DenseElementsAttr from a String attribute.
DenseElementsAttr createDenseElementsAttrFromStringAttrs(
    PatternRewriter &rewriter, Type elementType, SmallVector<Attribute> attrs) {
  SmallVector<int64_t, 1> dims(1, attrs.size());
  SmallVector<StringRef, 1> values;
  for (auto attr : attrs) {
    values.push_back(attr.cast<StringAttr>().getValue());
  }
  auto tensorType = mlir::RankedTensorType::get(dims, elementType);
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
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
  auto tensorType =
      mlir::RankedTensorType::get(dims, rewriter.getIntegerType(64));
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
}

// Create a DenseElementsAttr based on the size of type.
DenseElementsAttr createDenseElementsAttrFromSize(
    PatternRewriter &rewriter, Value value) {
  auto inType = value.getType().cast<ShapedType>();
  SmallVector<int64_t, 1> dims(1, 1);
  SmallVector<int64_t, 1> values = {inType.getNumElements()};
  auto tensorType =
      mlir::RankedTensorType::get(dims, rewriter.getIntegerType(64));
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
}

/// Check whether a value is produced by a dense ONNXConstantOp.
bool isDenseONNXConstant(Value result) {
  Operation *op = result.getDefiningOp();

  ONNXConstantOp constOp = llvm::dyn_cast_or_null<ONNXConstantOp>(op);
  // Not a constant.
  if (!constOp)
    return false;

  // If the dense attribute is null, there must be buffer_id
  // attribute.
  if (!(op->getAttrOfType<::mlir::Attribute>("value")))
    return false;
  // The other attributes must be null.
  if (op->getAttrOfType<::mlir::Attribute>("sparse_value"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_float"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_floats"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_int"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_ints"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_string"))
    return false;
  if (op->getAttrOfType<::mlir::Attribute>("value_strings"))
    return false;

  return true;
}

/// Check if a value is a 16, 32 or 64 bit integer.
bool isCommonInteger(mlir::RankedTensorType tensorType) {
  return tensorType.getElementType().isInteger(16) ||
         tensorType.getElementType().isInteger(32) ||
         tensorType.getElementType().isInteger(64);
}

/// Get scalar value when it is a constant.
double getScalarValue(
    mlir::ONNXConstantOp constantOp, mlir::RankedTensorType tensorType) {
  double value;
  DenseElementsAttr attr = constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
  if (!attr)
    constantOp.emitError("DenseElementsAttr expected");
  if (isCommonInteger(tensorType)) {
    auto valueIt = attr.getValues<IntegerAttr>().begin();
    value = (double)(*valueIt).cast<IntegerAttr>().getInt();
  } else if (tensorType.getElementType().isF32()) {
    auto valueIt = attr.getFloatValues().begin();
    value = (double)(*valueIt).convertToFloat();
  } else if (tensorType.getElementType().isF64()) {
    auto valueIt = attr.getFloatValues().begin();
    value = (double)(*valueIt).convertToDouble();
  } else {
    llvm_unreachable("Unexpected type.");
  }
  return value;
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

// Canonicalize ONNXDepthToSpaceOp.
Value canonicalizeDepthToSpace(PatternRewriter &rewriter, Operation *op,
    Value input, IntegerAttr blocksizeAttr, StringAttr modeAttr) {
  assert(op && isa<ONNXDepthToSpaceOp>(op) && "Expecting a ONNXDepthToSpaceOp");

  ShapedType inputType = input.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t B = inputShape[0];
  int64_t C = inputShape[1];
  int64_t H = inputShape[2];
  int64_t W = inputShape[3];
  int64_t bs = blocksizeAttr.getSInt();
  StringRef mode = modeAttr.getValue();

  SmallVector<int64_t, 6> shape1;
  ArrayAttr perm;
  if (mode == "DCR") {
    shape1 = {B, bs, bs, C / (bs * bs), H, W};
    perm = rewriter.getI64ArrayAttr({0, 3, 4, 1, 5, 2});
  } else {
    assert(mode == "CRD" && "Unexpected mode");
    shape1 = {B, C / (bs * bs), bs, bs, H, W};
    perm = rewriter.getI64ArrayAttr({0, 1, 4, 2, 5, 3});
  }

  // DCR: reshape = onnx.Reshape(input, [B, bs, bs, C / (bs * bs), H, W])
  // CRD: reshape = onnx.Reshape(input, [B, C / (bs * bs), bs, bs, H, W])
  auto shapeConstantOp1 = getONNXConstantOpFromDenseAttr(
      rewriter, op->getLoc(), rewriter.getI64TensorAttr(shape1));
  auto reshape = rewriter
                     .create<ONNXReshapeOp>(op->getLoc(),
                         RankedTensorType::get(shape1, elementType), input,
                         shapeConstantOp1)
                     .getResult();

  // DCR: transpose = onnx.Transpose(reshape, [0, 3, 4, 1, 5, 2])
  // CRD: transpose = onnx.Transpose(reshape, [0, 1, 4, 2, 5, 3])
  SmallVector<int64_t, 6> transposeShape = {B, C / (bs * bs), H, bs, W, bs};
  auto transpose =
      rewriter
          .create<ONNXTransposeOp>(op->getLoc(),
              RankedTensorType::get(transposeShape, elementType), reshape, perm)
          .getResult();

  // res = onnx.Reshape(transpose, [B, C / (bs * bs), H * bs, W * bs])
  SmallVector<int64_t, 4> shape2 = {B, C / (bs * bs), H * bs, W * bs};
  auto shapeConstantOp2 = getONNXConstantOpFromDenseAttr(
      rewriter, op->getLoc(), rewriter.getI64TensorAttr(shape2));
  return rewriter.create<ONNXReshapeOp>(op->getLoc(),
      RankedTensorType::get(shape2, elementType), transpose, shapeConstantOp2);
}
