/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ONNXOpsHelper.cpp - Helper functions for ONNX dialects -------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for lowering ONNX ops to Krnl Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
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
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    if (ttp.getEncoding().dyn_cast_or_null<ONNXTensorEncodingAttr>())
      return true;
  return false;
}

ONNXTensorEncodingAttr getONNXTensorEncoding(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<ONNXTensorEncodingAttr>();
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
  auto type1 = dyn_cast_or_null<ShapedType>(tensorOrMemref1.getType());
  auto type2 = dyn_cast_or_null<ShapedType>(tensorOrMemref2.getType());
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
  SmallVector<IndexExpr, 2> kernelExprs = {LiteralIndexExpr(0), start2};
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

size_t ArrayAttrSize(Optional<ArrayAttr> a) { return a.value().size(); }

int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
  return (a.getValue()[i]).cast<IntegerAttr>().getInt();
}

int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i) {
  return (a.value().getValue()[i]).cast<IntegerAttr>().getInt();
}

void ArrayAttrIntVals(ArrayAttr a, mlir::SmallVectorImpl<int64_t> &i) {
  for (size_t k = 0; k < a.size(); ++k)
    i.emplace_back((a.getValue()[k]).cast<IntegerAttr>().getInt());
}

ElementsAttr getElementAttributeFromONNXValue(Value value) {
  ONNXConstantOp constantOp = getONNXConstantOp(value);
  if (constantOp)
    return constantOp.getValueAttr().dyn_cast<ElementsAttr>();
  return nullptr;
}

// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value) {
  return dyn_cast_or_null<ONNXConstantOp>(value.getDefiningOp());
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
  if (!permAttr)
    return false;
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
  ElementsAttr shapeAttr = getElementAttributeFromONNXValue(shape);
  if (shapeAttr == nullptr)
    return false;

  int64_t dimensionsOfShape = shapeAttr.getShapedType().getShape()[0];
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

/// Test if 'val' has shape and rank or not.
bool hasShapeAndRank(Value val) {
  Type valType = val.getType();
  ShapedType shapedType;
  if (SeqType seqType = valType.dyn_cast<SeqType>())
    shapedType = seqType.getElementType().dyn_cast<ShapedType>();
  else if (OptType optType = valType.dyn_cast<OptType>())
    shapedType = optType.getElementType().dyn_cast<ShapedType>();
  else
    shapedType = valType.dyn_cast<ShapedType>();
  return shapedType && shapedType.hasRank();
}

bool hasShapeAndRank(Operation *op) {
  int num = op->getNumOperands();
  for (int i = 0; i < num; ++i)
    if (!hasShapeAndRank(op->getOperand(i)))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Support for rewrite patterns.
//===----------------------------------------------------------------------===//

// Create an ArrayAttr from a dense ConstantOp
ArrayAttr createArrayAttrFromConstantOp(ONNXConstantOp constOp) {
  auto elements = cast<ElementsAttr>(constOp.getValueAttr());
  SmallVector<Attribute> values(elements.getValues<Attribute>());
  return ArrayAttr::get(constOp.getContext(), values);
}

// Create a DenseElementsAttr from a float attribute.
DenseElementsAttr createDenseElementsAttrFromFloatAttr(
    PatternRewriter &rewriter, Type elementType, FloatAttr attr) {
  auto tensorType = RankedTensorType::get({1}, elementType);
  return DenseElementsAttr::get(tensorType, {attr.getValue()});
}

//===----------------------------------------------------------------------===//
// Support for dim operations.
//===----------------------------------------------------------------------===//

/// Check the defining operation of a value.
template <typename OP>
bool definedBy(Value v) {
  return !v.isa<BlockArgument>() && isa<OP>(v.getDefiningOp());
}

/// Template instantiation for definedBy.
template bool definedBy<ONNXCastOp>(Value v);
template bool definedBy<ONNXConcatOp>(Value v);
template bool definedBy<ONNXConstantOp>(Value v);
template bool definedBy<ONNXDimOp>(Value v);

/// Check if a value is to store dimensions, meaning it is defined by
/// Dim/Constant/Cast/Concat.
bool areDims(Value val) {
  // Value must be a 1D tensor.
  Type vType = val.getType();
  if (!(isRankedShapedType(vType) && (getRank(vType) == 1)))
    return false;

  // Base case.
  if (definedBy<ONNXConstantOp>(val) || definedBy<ONNXDimOp>(val) ||
      definedBy<ONNXCastOp>(val)) {
    // Value must be a 1D tensor of one element.
    return (getShape(vType)[0] == 1);
  }

  // Recursion case.
  if (definedBy<ONNXConcatOp>(val)) {
    // Recursively check.
    for (Value v : val.getDefiningOp()->getOperands())
      if (!areDims(v))
        return false;
    return true;
  }

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

Value normalizeConstantOp(
    PatternRewriter &rewriter, Value output, Attribute attr) {
  ShapedType outputType = output.getType().cast<ShapedType>();
  Type elementType = outputType.getElementType();

  DenseElementsAttr denseAttr;
  if (ArrayAttr arrayAttr = attr.dyn_cast<ArrayAttr>()) {
    int64_t dim = arrayAttr.size();
    auto tensorType = RankedTensorType::get({dim}, elementType);
    denseAttr = DenseElementsAttr::get(tensorType, arrayAttr.getValue());
  } else {
    auto tensorType = RankedTensorType::get({}, elementType);
    if (FloatAttr floatAttr = attr.dyn_cast<FloatAttr>()) {
      denseAttr = DenseElementsAttr::get(tensorType, {floatAttr.getValue()});
    } else if (IntegerAttr intAttr = attr.dyn_cast<IntegerAttr>()) {
      denseAttr = DenseElementsAttr::get(tensorType, intAttr.getSInt());
    } else if (StringAttr strAttr = attr.dyn_cast<StringAttr>()) {
      denseAttr = DenseElementsAttr::get(tensorType, {strAttr.getValue()});
    } else {
      llvm_unreachable("unexpected Attribute");
    }
  }
  return rewriter.create<ONNXConstantOp>(output.getLoc(), output.getType(),
      Attribute(), denseAttr, FloatAttr(), ArrayAttr(), IntegerAttr(),
      ArrayAttr(), StringAttr(), ArrayAttr());
}

// Create a DenseElementsAttr based on the shape of type at the given index.
DenseElementsAttr createDenseElementsAttrFromShapeAtIndex(
    PatternRewriter &rewriter, Value value, IntegerAttr indexAttr) {
  auto inType = value.getType().cast<ShapedType>();
  ArrayRef<int64_t> shape = inType.getShape();
  int64_t index = indexAttr.getValue().getSExtValue();
  SmallVector<int64_t, 4> values(1, shape[index]);
  auto tensorType = RankedTensorType::get({1}, rewriter.getIntegerType(64));
  return DenseElementsAttr::get(tensorType, ArrayRef(values));
}

// Create a DenseElementsAttr based on the size of type.
DenseElementsAttr createDenseElementsAttrFromSize(
    PatternRewriter &rewriter, Value value) {
  auto inType = value.getType().cast<ShapedType>();
  // Output Type should be scalar: tensor<i64>
  SmallVector<int64_t, 1> dims;
  SmallVector<int64_t, 1> values = {inType.getNumElements()};
  auto tensorType = RankedTensorType::get(dims, rewriter.getIntegerType(64));
  return DenseElementsAttr::get(tensorType, ArrayRef(values));
}

/// Check whether a value is produced by a dense ONNXConstantOp.
bool isDenseONNXConstant(Value result) {
  ONNXConstantOp constOp =
      dyn_cast_or_null<ONNXConstantOp>(result.getDefiningOp());

  // Must be a constant.
  if (!constOp)
    return false;

  // The value attribute must be an ElementsAttr (which is one of
  // DenseElementsAttr, DenseResourceElementsAttr, DisposableElementsAttr).
  if (!isa_and_nonnull<ElementsAttr>(constOp.getValueAttr()))
    return false;

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
RESULT_TYPE getScalarValue(ONNXConstantOp constantOp) {
  Type type = constantOp.getType();
  ElementsAttr attr = constantOp.getValueAttr().dyn_cast<ElementsAttr>();
  if (!attr)
    constantOp.emitError("ElementsAttr expected");
  return getScalarValue<RESULT_TYPE>(attr, type);
}

// Template instantiation for getScalarValue

template double getScalarValue<double>(ONNXConstantOp constantOp);
template int64_t getScalarValue<int64_t>(ONNXConstantOp constantOp);

// Convert type to MLIR type.
// A complete list of types can be found in:
// <onnx-mlir-build-folder>/third_party/onnx/onnx/onnx.pb.h
// TODO: Update Int*/Uint* to emit signed/unsigned MLIR types
Type convertONNXTypeToMLIRType(
    OpBuilder &builder_, onnx::TensorProto_DataType onnxType) {
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
    return ONNXStringType::get(builder_.getContext());

  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FNUZ:
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
  case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2FNUZ:
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
  if (elemType.isa<ONNXStringType>() ||
      elemType.isa<onnx_mlir::krnl::StringType>())
    return onnx::TensorProto::STRING;

  TypeSwitch<Type>(elemType)
      .Case<BFloat16Type>(
          [&](BFloat16Type) { onnxType = onnx::TensorProto::BFLOAT16; })
      .Case<ComplexType>([&](ComplexType type) {
        if (type.getElementType().isa<Float32Type>())
          onnxType = onnx::TensorProto::COMPLEX64;
        else if (type.getElementType().isa<Float64Type>())
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

} // namespace onnx_mlir
