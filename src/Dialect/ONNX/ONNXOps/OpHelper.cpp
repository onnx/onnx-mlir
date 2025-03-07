/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ONNXOpsHelper.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Path.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXLayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

// Identity affine
using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX Tensor support.

/// Get a ONNX Tensor data layout by StringRef. If layout string is a standard
/// layout, or any other unrecognized string, just return false.
bool convertStringToONNXCustomTensorDataLayout(StringAttr layoutAttr,
    ONNXTensorEncodingAttr::DataLayout &layout, int64_t &xFactor,
    int64_t &yFactor) {
  StringRef layoutStr(layoutAttr.getValue());
  if (layoutStr.equals_insensitive(LAYOUT_NCHW4C)) {
    xFactor = 4;
    yFactor = 0;
    layout = ONNXTensorEncodingAttr::DataLayout::NCHWxC;
    return true;
  } else if (layoutStr.equals_insensitive(LAYOUT_KCMN4C4K)) {
    xFactor = yFactor = 4;
    layout = ONNXTensorEncodingAttr::DataLayout::KCNMxCyK;
    return true;
  } else if (layoutStr.equals_insensitive(LAYOUT_STANDARD)) {
    // Represent standard layout by no layout, return false.
    // We really should not get there, but there is no harm in doing so.
    return false;
  }
  llvm_unreachable("unknown ONNX Tensor Data Layout");
}

/// Convert a data layout to StringRef.
StringRef convertONNXTensorDataLayoutToString(
    ONNXTensorEncodingAttr::DataLayout layout, int64_t xFactor,
    int64_t yFactor) {
  switch (layout) {
  case ONNXTensorEncodingAttr::DataLayout::NCHWxC:
    if (xFactor == 4 && yFactor == 0)
      return StringRef(LAYOUT_NCHW4C);
    llvm_unreachable("NCHWxC with unsupported x or y factors");
    break;
  case ONNXTensorEncodingAttr::DataLayout::KCNMxCyK:
    if (xFactor == 4 && yFactor == 4)
      return StringRef(LAYOUT_KCMN4C4K);
    llvm_unreachable("KCNMxCyK with unsupported x or y factors");
    break;
  case ONNXTensorEncodingAttr::DataLayout::STANDARD:
    if (xFactor == 0 && yFactor == 0)
      return StringRef(LAYOUT_STANDARD);
    llvm_unreachable("Standard with unsupported x or y factors");
  }
  llvm_unreachable("unsupported ONNX Layout");
}

bool isONNXTensor(const Type type) {
  if (auto ttp = mlir::dyn_cast<RankedTensorType>(type))
    if (mlir::dyn_cast_or_null<ONNXTensorEncodingAttr>(ttp.getEncoding()))
      return true;
  return false;
}

ONNXTensorEncodingAttr getONNXTensorEncoding(Type type) {
  if (auto ttp = mlir::dyn_cast<RankedTensorType>(type))
    return mlir::dyn_cast_or_null<ONNXTensorEncodingAttr>(ttp.getEncoding());
  return nullptr;
}

ONNXTensorEncodingAttr::DataLayout getONNXTensorLayout(Type type) {
  if (ONNXTensorEncodingAttr encoding = getONNXTensorEncoding(type))
    return encoding.getDataLayout();
  return ONNXTensorEncodingAttr::DataLayout::STANDARD;
}

// Return true if both types have the same ONNX Tensor Data Layout (does not
// check for dimensions, elementary types...).
bool identicalONNXTensorDataLayout(const Type type1, const Type type2) {

  ONNXTensorEncodingAttr encoding1 = getONNXTensorEncoding(type1);
  ONNXTensorEncodingAttr encoding2 = getONNXTensorEncoding(type2);
  // Test if neither have encodings, then it is considered identical.
  if (!encoding1 && !encoding2)
    return true;
  // Have encoding, test that they have the same parameters
  ONNXTensorEncodingAttr::DataLayout layout1 = encoding1.getDataLayout();
  ONNXTensorEncodingAttr::DataLayout layout2 = encoding2.getDataLayout();
  return layout1 == layout2 &&
         encoding1.getXFactor() == encoding2.getXFactor() &&
         encoding1.getYFactor() == encoding2.getYFactor();
}

bool hasConvONNXTensorDataLayout(const Type type) {
  ONNXTensorEncodingAttr::DataLayout layout = getONNXTensorLayout(type);
  return (layout == ONNXTensorEncodingAttr::DataLayout::NCHWxC ||
          layout == ONNXTensorEncodingAttr::DataLayout::KCNMxCyK);
}

bool hasCustomONNXTensorDataLayout(const Type type) {
  return getONNXTensorLayout(type) !=
         ONNXTensorEncodingAttr::DataLayout::STANDARD;
}

bool sameRank(Value tensorOrMemref1, Value tensorOrMemref2) {
  auto type1 = mlir::dyn_cast_or_null<ShapedType>(tensorOrMemref1.getType());
  auto type2 = mlir::dyn_cast_or_null<ShapedType>(tensorOrMemref2.getType());
  if (!type1 || !type2)
    return false;
  if (!type1.hasRank() || !type2.hasRank())
    return false;
  return (type1.getRank() == type2.getRank());
}

// Add a tensor encoding to a rank & shaped type. Otherwise, return an unranked
// type as it is.
Type convertTensorTypeToTensorTypeWithEncoding(
    const Type inputType, Attribute encodingAttr) {
  Type resType = inputType;
  if (auto rankedType = llvm::dyn_cast_or_null<RankedTensorType>(inputType)) {
    // Compute shape: this op does not change the shape, just the layout.
    ArrayRef<int64_t> inputShape = rankedType.getShape();
    SmallVector<int64_t, 4> resShape(inputShape.begin(), inputShape.end());
    resType = RankedTensorType::get(
        resShape, rankedType.getElementType(), encodingAttr);
  }
  return resType;
}

//===----------------------------------------------------------------------===//

AffineMap getIdentityDimMap(Builder &builder) {
  return AffineMap::get(1, 0, {builder.getAffineDimExpr(0)});
}
//===----------------------------------------------------------------------===//
// ONNX Pool Conv Support.

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
/// The conv/pooling window can be smaller than the kernel when slicing it
/// over the border edges. Thus, we will compute the start and end indices for
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
///   When dilation is non-unit, the first valid pixel to apply conv/pooling
///   on will not be the 0-th pixel, but rather the smallest integer n to make
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
  SmallVector<IndexExpr, 2> kernelExprs = {LitIE(0), start2};
  kernelOffsetExpr = IndexExpr::min(kernelExprs);

  return std::vector<IndexExpr>{
      windowStartExpr, windowEndExpr, kernelOffsetExpr};
}

/// The conv/pooling window can be smaller than the kernel when slicing it
/// over the border edges. This function returns an AffineMap to compute the
/// size of one edge of the window.
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
  // Upper bound for an affine.for is `min AffineMap`, where `min` is
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

size_t ArrayAttrSize(std::optional<ArrayAttr> a) { return a.value().size(); }

int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
  return mlir::cast<IntegerAttr>(a.getValue()[i]).getInt();
}

int64_t ArrayAttrIntVal(std::optional<ArrayAttr> a, int i) {
  return mlir::cast<IntegerAttr>(a.value().getValue()[i]).getInt();
}

void ArrayAttrIntVals(ArrayAttr a, mlir::SmallVectorImpl<int64_t> &i) {
  for (size_t k = 0; k < a.size(); ++k)
    i.emplace_back(mlir::cast<IntegerAttr>(a.getValue()[k]).getInt());
}

ElementsAttr getElementAttributeFromONNXValue(Value value) {
  ONNXConstantOp constantOp = getONNXConstantOp(value);
  // In case the ConstantOp has not been normalized yet
  if (constantOp && constantOp.getValueAttr())
    return mlir::dyn_cast<ElementsAttr>(constantOp.getValueAttr());
  return nullptr;
}

// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value) {
  return mlir::dyn_cast_or_null<ONNXConstantOp>(value.getDefiningOp());
}

bool getI64ValuesFromONNXConstantOp(
    Value val, mlir::SmallVectorImpl<int64_t> &iRes) {
  ElementsAttr elemsAttr = getElementAttributeFromONNXValue(val);
  if (!elemsAttr)
    return false;
  if (!getElementType(elemsAttr.getType()).isInteger(64))
    return false;
  SmallVector<int64_t, 4> iVals(elemsAttr.getValues<int64_t>());
  iRes.append(iVals);
  return true;
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
    initialPerm.emplace_back(mlir::cast<IntegerAttr>(firstPermVal).getInt());
  // Read second permute vector. Use it as an index in the first permute
  // vector.
  SmallVector<int64_t, 4> resPerm;
  for (auto secondPermVal : secondPermAttr.getValue()) {
    auto index = mlir::cast<IntegerAttr>(secondPermVal).getInt();
    resPerm.emplace_back(initialPerm[index]);
  }
  // Convert to Array of Attributes.
  ArrayRef<int64_t> resPermRefs(resPerm);
  return rewriter.getI64ArrayAttr(resPermRefs);
}

/// Test if the permute pattern correspond to an identity pattern.
/// Identity patterns are {0, 1, 2, ... , rank -1}.
bool IsIdentityPermuteVector(ArrayAttr permAttr) {
  if (!permAttr)
    return false;
  int64_t currentIndex = 0;
  for (auto permVal : permAttr.getValue())
    if (mlir::cast<IntegerAttr>(permVal).getInt() != currentIndex++)
      return false;
  return true;
}

/// Test if the value has the specified constant shape
bool HasSpecifiedConstantShape(Value value, Value shape) {
  if (!hasShapeAndRank(value) || !hasShapeAndRank(shape))
    return false;

  ArrayRef<int64_t> valueShape =
      mlir::cast<ShapedType>(value.getType()).getShape();
  ElementsAttr shapeAttr = getElementAttributeFromONNXValue(shape);
  if (shapeAttr == nullptr)
    return false;

  int64_t dimensionsOfShape = shapeAttr.getShapedType().getShape()[0];
  if (static_cast<int64_t>(valueShape.size()) != dimensionsOfShape)
    return false;

  auto valueIt = shapeAttr.getValues<APInt>().begin();
  for (int64_t i = 0; i < dimensionsOfShape; i++) {
    int64_t value = (*valueIt++).getSExtValue();
    if (valueShape[i] != value)
      return false;
  }
  return true;
}

/// Test if a value is a scalar constant tensor or not, i.e. tensor<dtype> or
/// tensor<1xdtype>.
bool isScalarConstantTensor(Value v) {
  if (!hasShapeAndRank(v))
    return false;

  auto t = mlir::dyn_cast<ShapedType>(v.getType());
  int64_t r = t.getRank();
  return isDenseONNXConstant(v) &&
         ((r == 0) || ((r == 1) && (t.getShape()[0] == 1)));
}

/// Test if 'val' has shape and rank or not.
bool hasShapeAndRank(Value val) {
  Type valType = val.getType();
  ShapedType shapedType;
  if (SeqType seqType = mlir::dyn_cast<SeqType>(valType))
    shapedType = mlir::dyn_cast<ShapedType>(seqType.getElementType());
  else if (OptType optType = mlir::dyn_cast<OptType>(valType))
    shapedType = mlir::dyn_cast<ShapedType>(optType.getElementType());
  else
    shapedType = mlir::dyn_cast<ShapedType>(valType);
  return shapedType && shapedType.hasRank();
}

bool hasShapeAndRank(Operation *op) {
  int num = op->getNumOperands();
  for (int i = 0; i < num; ++i)
    if (!hasShapeAndRank(op->getOperand(i)))
      return false;
  return true;
}

/// Test if a value has only one use except ONNXDimOp.
bool hasOneUseExceptDimOp(Value val) {
  int64_t numOfUsersExceptDim = 0;
  for (auto user : val.getUsers()) {
    if (isa<ONNXDimOp>(user))
      continue;
    numOfUsersExceptDim++;
  }
  return (numOfUsersExceptDim == 1);
}

//===----------------------------------------------------------------------===//
// Support for rewrite patterns.
//===----------------------------------------------------------------------===//

// Create an ArrayAttr from a dense ConstantOp
ArrayAttr createArrayAttrFromConstantOp(ONNXConstantOp constOp) {
  auto elements = mlir::cast<ElementsAttr>(constOp.getValueAttr());
  SmallVector<Attribute> values(elements.getValues<Attribute>());
  return ArrayAttr::get(constOp.getContext(), values);
}

// Create a DenseElementsAttr from a float attribute.
DenseElementsAttr createDenseElementsAttrFromFloatAttr(
    PatternRewriter &rewriter, Type elementType, FloatAttr attr) {
  auto tensorType = RankedTensorType::get({1}, elementType);
  auto ftype = mlir::cast<FloatType>(elementType);
  APFloat f = attr.getValue();
  bool ignored;
  f.convert(ftype.getFloatSemantics(), APFloat::rmNearestTiesToEven, &ignored);
  return DenseElementsAttr::get(tensorType, {f});
}

//===----------------------------------------------------------------------===//
// Support for dim operations.
//===----------------------------------------------------------------------===//

/// Check if a value is to store dimensions, meaning it is a tensor of one
/// element or concatenation of one-element tensors.
bool areDims(Value val) {
  // Value must be a 1D tensor.
  Type vType = val.getType();
  if (!(isRankedShapedType(vType) && (getRank(vType) == 1)))
    return false;

  // Recursion case.
  if (definedBy<ONNXConcatOp>(val)) {
    // Recursively check.
    for (Value v : val.getDefiningOp()->getOperands())
      if (!areDims(v))
        return false;
    return true;
  }

  // Base case.
  // A dimension must be a 1D tensor of one i64 element.
  if ((getShape(vType)[0] == 1) && getElementType(vType).isSignlessInteger(64))
    return true;

  // Not Dim/Constant/Cast/Concat.
  return false;
}

/// Check if a value is defined by Concat to store dimensions.
bool areDimsFromConcat(Value val) {
  return (areDims(val) && definedBy<ONNXConcatOp>(val));
}

/// Get all dimensions that are stored by the value.
void getDims(Value val, SmallVectorImpl<Value> &dims) {
  assert(areDims(val) && "Value does not store dimensions");
  if (definedBy<ONNXConcatOp>(val)) {
    for (Value v : val.getDefiningOp()->getOperands()) {
      SmallVector<Value, 4> inputs;
      getDims(v, inputs);
      for (Value i : inputs)
        dims.emplace_back(i);
    }
  } else
    dims.emplace_back(val);
}

// Create a DenseElementsAttr based on the shape of type at the given index.
DenseElementsAttr createDenseElementsAttrFromShapeAtIndex(
    PatternRewriter &rewriter, Value value, IntegerAttr indexAttr) {
  auto inType = mlir::cast<ShapedType>(value.getType());
  ArrayRef<int64_t> shape = inType.getShape();
  int64_t index = indexAttr.getValue().getSExtValue();
  SmallVector<int64_t, 4> values(1, shape[index]);
  auto tensorType = RankedTensorType::get({1}, rewriter.getIntegerType(64));
  return DenseElementsAttr::get(tensorType, ArrayRef(values));
}

// Create a DenseElementsAttr based on the size of type.
DenseElementsAttr createDenseElementsAttrFromSize(
    PatternRewriter &rewriter, Value value) {
  auto inType = mlir::cast<ShapedType>(value.getType());
  // Output Type should be scalar: tensor<i64>
  SmallVector<int64_t, 1> dims;
  SmallVector<int64_t, 1> values = {inType.getNumElements()};
  auto tensorType = RankedTensorType::get(dims, rewriter.getIntegerType(64));
  return DenseElementsAttr::get(tensorType, ArrayRef(values));
}

/// Check whether a value is produced by a dense ONNXConstantOp.
bool isDenseONNXConstant(Value result) {
  ONNXConstantOp constOp =
      mlir::dyn_cast_or_null<ONNXConstantOp>(result.getDefiningOp());

  // Must be a constant.
  if (!constOp)
    return false;

  // Must have value attribute.
  Attribute value = constOp.getValueAttr();
  if (!value)
    return false;

  assert((isa<DenseElementsAttr, DisposableElementsAttr>(value)) &&
         "unsupported onnx constant value attribute");

  // No other attribute must be set.
  return !constOp.getValueFloatAttr() && !constOp.getValueFloatsAttr() &&
         !constOp.getValueIntAttr() && !constOp.getValueIntsAttr() &&
         !constOp.getValueStringAttr() && !constOp.getValueStringsAttr() &&
         !constOp.getSparseValueAttr();
}

/// Get scalar value when it is a constant.
template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(ElementsAttr denseAttr, Type type) {
  Type elementaryType = getElementTypeOrSelf(type);
  if (elementaryType.isInteger(16) || elementaryType.isInteger(32) ||
      elementaryType.isInteger(64)) {
    auto valueIt = denseAttr.getValues<IntegerAttr>().begin();
    return static_cast<RESULT_TYPE>(mlir::cast<IntegerAttr>(*valueIt).getInt());
  } else if (mlir::isa<FloatType>(elementaryType)) {
    auto valueIt = denseAttr.getValues<APFloat>().begin();
    return static_cast<RESULT_TYPE>((*valueIt).convertToDouble());
  }
  llvm_unreachable("Unexpected type.");
  return 0;
}

template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(ONNXConstantOp constantOp) {
  Type type = constantOp.getType();
  ElementsAttr attr = mlir::dyn_cast<ElementsAttr>(constantOp.getValueAttr());
  if (!attr)
    constantOp.emitError("ElementsAttr expected");
  return getScalarValue<RESULT_TYPE>(attr, type);
}

// Template instantiation for getScalarValue

template double getScalarValue<double>(ONNXConstantOp constantOp);
template int64_t getScalarValue<int64_t>(ONNXConstantOp constantOp);

/// Return the wide type of a value.
WideNum asWideNum(double n, Type elemType) {
  return wideZeroDispatch(elemType, [n](auto wideZero) {
    using cpptype = decltype(wideZero);
    constexpr BType TAG = toBType<cpptype>;
    return WideNum::widen<TAG>(static_cast<cpptype>(n));
  });
}

/// Checks whether a constant tensor's elements are all equal to a given scalar.
bool isConstOf(Value constValue, double n) {
  ElementsAttr constElements = getElementAttributeFromONNXValue(constValue);
  Type elemType = constElements.getElementType();
  assert(!elemType.isInteger(1) && "booleans are not supported");
  WideNum w = asWideNum(n, elemType);
  return ElementsAttrBuilder::allEqual(constElements, w);
}

// Convert type to MLIR type.
// A complete list of types can be found in:
// <onnx-mlir-build-folder>/third_party/onnx/onnx/onnx.pb.h
// TODO: Update Int*/Uint* to emit signed/unsigned MLIR types
Type convertONNXTypeToMLIRType(
    Builder &builder, onnx::TensorProto_DataType onnxType) {
  switch (onnxType) {
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
    return builder.getFloat8E4M3FNType();
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FNUZ:
    return builder.getFloat8E4M3FNUZType();
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
    return builder.getFloat8E5M2Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2FNUZ:
    return builder.getFloat8E5M2FNUZType();
  case onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16:
    return builder.getBF16Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    return builder.getF16Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
    return builder.getF32Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
    return builder.getF64Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
    return builder.getIntegerType(/*width=*/8);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
    return builder.getIntegerType(/*width=*/8, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
    return builder.getIntegerType(/*width=*/16);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
    return builder.getIntegerType(/*width=*/16, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
    return builder.getIntegerType(/*width=*/32);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
    return builder.getIntegerType(/*width=*/32, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
    return builder.getIntegerType(/*width=*/64);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
    return builder.getIntegerType(/*width=*/64, false);
  case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
    return builder.getI1Type();
  case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
    return ONNXStringType::get(builder.getContext());
  case onnx::TensorProto_DataType::TensorProto_DataType_INT4:
    return builder.getIntegerType(/*width=*/4);
  case onnx::TensorProto_DataType::TensorProto_DataType_UINT4:
    return builder.getIntegerType(/*width=*/4, false);

  case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX64:
  case onnx::TensorProto_DataType::TensorProto_DataType_COMPLEX128:
  case onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED:
    llvm_unreachable("Unsupported data type encountered.");
    return nullptr;
  }

  llvm_unreachable("Unsupported data type encountered.");
}

// Convert an MLIR type to the corresponding ONNX type.
int64_t mlirTypeToOnnxType(Type elemType) {
  onnx::TensorProto::DataType onnxType = onnx::TensorProto::UNDEFINED;
  if (!elemType)
    return onnxType;

  TypeSwitch<Type>(elemType)
      .Case<ONNXStringType>(
          [&](ONNXStringType) { onnxType = onnx::TensorProto::STRING; })
      .Case<Float8E4M3FNType>(
          [&](Float8E4M3FNType) { onnxType = onnx::TensorProto::FLOAT8E4M3FN; })
      .Case<Float8E4M3FNUZType>([&](Float8E4M3FNUZType) {
        onnxType = onnx::TensorProto::FLOAT8E4M3FNUZ;
      })
      .Case<Float8E5M2Type>(
          [&](Float8E5M2Type) { onnxType = onnx::TensorProto::FLOAT8E5M2; })
      .Case<Float8E5M2FNUZType>([&](Float8E5M2FNUZType) {
        onnxType = onnx::TensorProto::FLOAT8E5M2FNUZ;
      })
      .Case<BFloat16Type>(
          [&](BFloat16Type) { onnxType = onnx::TensorProto::BFLOAT16; })
      .Case<ComplexType>([&](ComplexType type) {
        if (mlir::isa<Float32Type>(type.getElementType()))
          onnxType = onnx::TensorProto::COMPLEX64;
        else if (mlir::isa<Float64Type>(type.getElementType()))
          onnxType = onnx::TensorProto::COMPLEX128;
      })
      .Case<Float16Type>(
          [&](Float16Type) { onnxType = onnx::TensorProto::FLOAT16; })
      .Case<Float32Type>(
          [&](Float32Type) { onnxType = onnx::TensorProto::FLOAT; })
      .Case<Float64Type>(
          [&](Float64Type) { onnxType = onnx::TensorProto::DOUBLE; })
      .Case<IntegerType>([&](IntegerType type) {
        switch (type.getWidth()) {
        case 1:
          // only a signless type can be a bool.
          onnxType = (type.isSigned() || type.isUnsigned())
                         ? onnx::TensorProto::UNDEFINED
                         : onnx::TensorProto::BOOL;
          break;
        case 4:
          onnxType = type.isUnsigned() ? onnx::TensorProto::UINT4
                                       : onnx::TensorProto::INT4;
          break;
        case 8:
          onnxType = type.isUnsigned() ? onnx::TensorProto::UINT8
                                       : onnx::TensorProto::INT8;
          break;
        case 16:
          onnxType = type.isUnsigned() ? onnx::TensorProto::UINT16
                                       : onnx::TensorProto::INT16;
          break;
        case 32:
          onnxType = type.isUnsigned() ? onnx::TensorProto::UINT32
                                       : onnx::TensorProto::INT32;
          break;
        case 64:
          onnxType = type.isUnsigned() ? onnx::TensorProto::UINT64
                                       : onnx::TensorProto::INT64;
          break;
        }
      })
      .Case<LLVM::LLVMStructType>(
          [&](LLVM::LLVMStructType) { onnxType = onnx::TensorProto::STRING; });

  if (onnxType == onnx::TensorProto::UNDEFINED) {
    elemType.dump();
    llvm_unreachable("MLIR type cannot be converted to ONNX type");
  }

  return onnxType;
}

bool isScalarTensor(Value v) {
  return (hasShapeAndRank(v) &&
          ((getRank(v.getType()) == 0) ||
              (getRank(v.getType()) == 1 && getShape(v.getType())[0] == 1)));
}

bool hasIntegerPowerExponent(ONNXPowOp *op, int64_t &exponentValue) {
  Value exponent = op->getY();
  ElementsAttr elementAttr = getElementAttributeFromONNXValue(exponent);
  if (!elementAttr)
    return false;
  if (elementAttr.getNumElements() != 1)
    return false;
  Type elementType = elementAttr.getElementType();
  if (mlir::isa<FloatType>(elementType)) {
    double floatVal = getScalarValue<double>(elementAttr, elementType);
    if (floatVal == ceil(floatVal)) {
      // We essentially have an integer value represented as a float.
      exponentValue = static_cast<int64_t>(floatVal);
      return true;
    }
  } else if (mlir::isa<IntegerType>(elementType)) {
    exponentValue = getScalarValue<int64_t>(elementAttr, elementType);
    return true;
  }
  // Other type, just fails.
  return false;
}

//===----------------------------------------------------------------------===//
// Support for ReshapeOp.
//===----------------------------------------------------------------------===//

// Return true if reshape does nothing, aka it returns the same as the input.
// Use dimAnalysis if provided.

bool isIdentityReshape(
    Value inputTensor, Value outputTensor, const DimAnalysis *dimAnalysis) {
  if (!hasShapeAndRank(inputTensor) || !hasShapeAndRank(outputTensor))
    return false;
  Type inputType = inputTensor.getType();
  Type outputType = outputTensor.getType();
  ArrayRef<int64_t> inputShape = getShape(inputType);
  ArrayRef<int64_t> outputShape = getShape(outputType);
  int64_t inputRank = inputShape.size();
  int64_t outputRank = outputShape.size();

  // Check if same rank.
  if (inputRank != outputRank)
    return false;

  // Check if same shape in the sense that both dimensions at the same index
  // must be both static or dynamic. Otherwise, written rules may fail with the
  // following error due to shape mismatched:
  // ```
  // error: failed to materialize conversion for result #0 of operation
  // 'onnx.Reshape' that remained live after conversion
  // ```
  if (inputShape != outputShape)
    return false;

  // Reshape is an identity if at least (N-1) out of N dimensions are equal. We
  // don't need to care about the different dimension, it is maybe because of
  // DimAnalysis failed to handle it.
  int nSameDims = 0;
  for (int64_t i = 0; i < inputRank; ++i) {
    if (inputShape[i] != ShapedType::kDynamic &&
        inputShape[i] == outputShape[i])
      nSameDims++;
    else if (dimAnalysis &&
             dimAnalysis->sameDim(inputTensor, i, outputTensor, i))
      nSameDims++;
  }
  // Its basically ok to miss one as it then must be equal.
  if (nSameDims >= inputRank - 1)
    return true;

  return false;
}

bool isIdentityReshape(
    ONNXReshapeOp reshapeOp, const DimAnalysis *dimAnalysis) {
  // Check if ranked and shaped.
  Value inputTensor = reshapeOp.getData();
  Value outputTensor = reshapeOp.getReshaped();
  return isIdentityReshape(inputTensor, outputTensor, dimAnalysis);
}

//===----------------------------------------------------------------------===//
// Support for location.
//===----------------------------------------------------------------------===//

// We may try to relate the node names generated by the instrumentation
// with the node names printed by opt-report. Thus it is key to keep the
// code that generates these node name in sync.
//
// The code are found here:
// 1) `matchAndRewrite` from `src/Conversion/KrnlToLLVM/KrnlInstrument.cpp`
// 2) `getNodeNameInPresenceOfOpt` from
//    `src/Dialect/ONNX/ONNXOps/OpHelper.cpp`

std::string getNodeNameInPresenceOfOpt(Operation *op, bool useFileLine) {
  auto getNameFromFileLineLoc = [](FileLineColLoc loc, std::string &name,
                                    std::string postfix = "") {
    std::string filename =
        llvm::sys::path::filename(loc.getFilename().str()).str();
    name += filename + ":" + std::to_string(loc.getLine()) + postfix;
  };

  StringAttr nodeName;
  // Try with op onnx_node_name attribute.
  nodeName = op->getAttrOfType<StringAttr>("onnx_node_name");
  if (nodeName) {
    return nodeName.str();
  }
  // Try with op location.
  Location loc = op->getLoc();
  if (auto nameLoc = mlir::dyn_cast<NameLoc>(loc)) {
    return nameLoc.getName().str();
  }
  if (auto fusedLoc = mlir::dyn_cast<FusedLoc>(loc)) {
    // Combine each location name and set it as nodeName.
    std::string name;
    for (Location locIt : fusedLoc.getLocations()) {
      if (auto nameLocIt = mlir::dyn_cast<NameLoc>(locIt))
        name += nameLocIt.getName().str() + "-";
      else if (useFileLine) {
        if (auto fileLineColLoc = mlir::dyn_cast<FileLineColLoc>(locIt)) {
          getNameFromFileLineLoc(fileLineColLoc, name, "-");
        }
      }
    }
    if (name.empty())
      name = "NOTSET";
    else
      name.pop_back(); // remove last "-"
    return name;
  }
  if (useFileLine) {
    if (auto fileLineColLoc = mlir::dyn_cast<FileLineColLoc>(loc)) {
      std::string name = "";
      getNameFromFileLineLoc(fileLineColLoc, name);
      return name;
    }
  }
  return "NOTSET";
}

//===----------------------------------------------------------------------===//
// Support for DenseElementsAttr.
//===----------------------------------------------------------------------===//

bool isElementAttrUninitializedDenseResource(ElementsAttr elementsAttr) {
  const auto denseResourceElementsAttr =
      mlir::dyn_cast<DenseResourceElementsAttr>(elementsAttr);
  return denseResourceElementsAttr &&
         !denseResourceElementsAttr.getRawHandle().getBlob();
}

} // namespace onnx_mlir
