/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- OpHelper.cpp - NNPA ZHigh Helper Functions ------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zhigh {

/// Check if a value type is ranked or unranked.
bool hasRankedType(Value val) {
  ShapedType shapedType = mlir::cast<ShapedType>(val.getType());
  return (shapedType && shapedType.hasRank());
}

/// Get a ztensor data layout by StringAttr.
ZTensorEncodingAttr::DataLayout convertStringAttrToZTensorDataLayout(
    StringAttr layoutAttr) {
  if (layoutAttr) {
    StringRef layoutStr = layoutAttr.getValue();
    if (layoutStr.equals_insensitive(LAYOUT_1D))
      return ZTensorEncodingAttr::DataLayout::_1D;
    else if (layoutStr.equals_insensitive(LAYOUT_2D))
      return ZTensorEncodingAttr::DataLayout::_2D;
    else if (layoutStr.equals_insensitive(LAYOUT_2DS))
      return ZTensorEncodingAttr::DataLayout::_2DS;
    else if (layoutStr.equals_insensitive(LAYOUT_3D))
      return ZTensorEncodingAttr::DataLayout::_3D;
    else if (layoutStr.equals_insensitive(LAYOUT_3DS))
      return ZTensorEncodingAttr::DataLayout::_3DS;
    else if (layoutStr.equals_insensitive(LAYOUT_4D))
      return ZTensorEncodingAttr::DataLayout::_4D;
    else if (layoutStr.equals_insensitive(LAYOUT_4DS))
      return ZTensorEncodingAttr::DataLayout::_4DS;
    else if (layoutStr.equals_insensitive(LAYOUT_NCHW))
      return ZTensorEncodingAttr::DataLayout::NCHW;
    else if (layoutStr.equals_insensitive(LAYOUT_NHWC))
      return ZTensorEncodingAttr::DataLayout::NHWC;
    else if (layoutStr.equals_insensitive(LAYOUT_HWCK))
      return ZTensorEncodingAttr::DataLayout::HWCK;
    else if (layoutStr.equals_insensitive(LAYOUT_FICO))
      return ZTensorEncodingAttr::DataLayout::FICO;
    else if (layoutStr.equals_insensitive(LAYOUT_ZRH))
      return ZTensorEncodingAttr::DataLayout::ZRH;
    else if (layoutStr.equals_insensitive(LAYOUT_BFICO))
      return ZTensorEncodingAttr::DataLayout::BFICO;
    else if (layoutStr.equals_insensitive(LAYOUT_BZRH))
      return ZTensorEncodingAttr::DataLayout::BZRH;
    else
      llvm_unreachable("Invalid data layout string");
  } else
    llvm_unreachable("Could not get layout by an empty StringAttr");
}

/// Get a ztensor data layout by rank.
ZTensorEncodingAttr::DataLayout getZTensorDataLayoutByRank(int64_t rank) {
  if (rank == 1)
    return ZTensorEncodingAttr::DataLayout::_1D;
  else if (rank == 2)
    return ZTensorEncodingAttr::DataLayout::_2D;
  else if (rank == 3)
    // Use 3DS instead of 3D since important ops like LSTM/MatMul/Softmax use
    // 3DS, which reduces the number of layout transformations.
    return ZTensorEncodingAttr::DataLayout::_3DS;
  else if (rank == 4)
    return ZTensorEncodingAttr::DataLayout::_4D;
  else
    llvm_unreachable(
        "Could not get layout by rank. Rank must be 1, 2, 3, or 4");
}

/// Convert a data layout to StringAttr.
StringAttr convertZTensorDataLayoutToStringAttr(
    OpBuilder &builder, ZTensorEncodingAttr::DataLayout layout) {
  StringAttr attr;
  switch (layout) {
  case ZTensorEncodingAttr::DataLayout::_1D:
    attr = builder.getStringAttr(LAYOUT_1D);
    break;
  case ZTensorEncodingAttr::DataLayout::_2D:
    attr = builder.getStringAttr(LAYOUT_2D);
    break;
  case ZTensorEncodingAttr::DataLayout::_2DS:
    attr = builder.getStringAttr(LAYOUT_2DS);
    break;
  case ZTensorEncodingAttr::DataLayout::_3D:
    attr = builder.getStringAttr(LAYOUT_3D);
    break;
  case ZTensorEncodingAttr::DataLayout::_3DS:
    attr = builder.getStringAttr(LAYOUT_3DS);
    break;
  case ZTensorEncodingAttr::DataLayout::_4D:
    attr = builder.getStringAttr(LAYOUT_4D);
    break;
  case ZTensorEncodingAttr::DataLayout::_4DS:
    attr = builder.getStringAttr(LAYOUT_4DS);
    break;
  case ZTensorEncodingAttr::DataLayout::NCHW:
    attr = builder.getStringAttr(LAYOUT_NCHW);
    break;
  case ZTensorEncodingAttr::DataLayout::NHWC:
    attr = builder.getStringAttr(LAYOUT_NHWC);
    break;
  case ZTensorEncodingAttr::DataLayout::HWCK:
    attr = builder.getStringAttr(LAYOUT_HWCK);
    break;
  case ZTensorEncodingAttr::DataLayout::FICO:
    attr = builder.getStringAttr(LAYOUT_FICO);
    break;
  case ZTensorEncodingAttr::DataLayout::BFICO:
    attr = builder.getStringAttr(LAYOUT_BFICO);
    break;
  case ZTensorEncodingAttr::DataLayout::ZRH:
    attr = builder.getStringAttr(LAYOUT_ZRH);
    break;
  case ZTensorEncodingAttr::DataLayout::BZRH:
    attr = builder.getStringAttr(LAYOUT_BZRH);
    break;
  default:
    break;
  }
  return attr;
}

/// Get a ztensor quantized type by StringAttr.
ZTensorEncodingAttr::QuantizedType convertStringAttrToZTensorQuantizedType(
    StringAttr qtypeAttr) {
  if (qtypeAttr) {
    StringRef qtypeStr = qtypeAttr.getValue();
    if (qtypeStr.equals_insensitive(QTYPE_DLFLOAT16))
      return ZTensorEncodingAttr::QuantizedType::DLFLOAT16;
    else if (qtypeStr.equals_insensitive(QTYPE_INT8))
      return ZTensorEncodingAttr::QuantizedType::INT8;
    else if (qtypeStr.equals_insensitive(QTYPE_WEIGHTS))
      return ZTensorEncodingAttr::QuantizedType::WEIGHTS;
    else if (qtypeStr.equals_insensitive(QTYPE_UNDEFINED))
      return ZTensorEncodingAttr::QuantizedType::UNDEFINED;
    else
      llvm_unreachable("Invalid quantized type string");
  } else
    llvm_unreachable("Could not get quantized type by an empty StringAttr");
}

/// Convert a quantized type to StringAttr.
StringAttr convertZTensorQuantizedTypeToStringAttr(
    OpBuilder &builder, ZTensorEncodingAttr::QuantizedType qtype) {
  StringAttr attr;
  switch (qtype) {
  case ZTensorEncodingAttr::QuantizedType::DLFLOAT16:
    attr = builder.getStringAttr(QTYPE_DLFLOAT16);
    break;
  case ZTensorEncodingAttr::QuantizedType::INT8:
    attr = builder.getStringAttr(QTYPE_INT8);
    break;
  case ZTensorEncodingAttr::QuantizedType::WEIGHTS:
    attr = builder.getStringAttr(QTYPE_WEIGHTS);
    break;
  default:
    break;
  }
  return attr;
}

//===----------------------------------------------------------------------===//
// Utility functions to query ztensor information.

bool isZTensor(Type type) {
  if (auto ttp = mlir::dyn_cast<RankedTensorType>(type))
    if (mlir::dyn_cast_or_null<ZTensorEncodingAttr>(ttp.getEncoding()))
      return true;
  return false;
}

ZTensorEncodingAttr getZTensorEncoding(Type type) {
  if (auto ttp = mlir::dyn_cast<RankedTensorType>(type))
    return mlir::dyn_cast_or_null<ZTensorEncodingAttr>(ttp.getEncoding());
  return nullptr;
}

ZTensorEncodingAttr::DataLayout getZTensorLayout(Type type) {
  if (auto encoding = getZTensorEncoding(type))
    return encoding.getDataLayout();
  return ZTensorEncodingAttr::DataLayout::UNDEFINED;
}

StringAttr getZTensorLayoutAttr(OpBuilder &builder, Type type) {
  ZTensorEncodingAttr::DataLayout layout = getZTensorLayout(type);
  if (layout != ZTensorEncodingAttr::DataLayout::UNDEFINED)
    return convertZTensorDataLayoutToStringAttr(builder, layout);
  return nullptr;
}

ZTensorEncodingAttr::QuantizedType getZTensorQuantizedType(Type type) {
  if (auto encoding = getZTensorEncoding(type))
    return encoding.getQuantizedType();
  return ZTensorEncodingAttr::QuantizedType::UNDEFINED;
}

//===----------------------------------------------------------------------===//
// Utility functions.

Value getMinusBcastConst(
    OpBuilder &builder, Location loc, FloatAttr floatAttr, Value X) {
  ShapedType xType = mlir::cast<ShapedType>(X.getType());
  assert(xType.hasStaticShape() && "expected static shape");
  float val = floatAttr.getValueAsDouble() * -1.0;
  DenseElementsAttr denseAttr =
      DenseElementsAttr::get(mlir::cast<ShapedType>(X.getType()), val);
  MultiDialectBuilder<OnnxBuilder> create(builder, loc);
  return create.onnx.constant(denseAttr);
}

Value getConstantOfType(
    OpBuilder &builder, Location loc, Type type, float val) {
  ShapedType shapedType = mlir::cast<ShapedType>(type);
  assert(shapedType.hasStaticShape() && "expected static shape");
  Type elementType = shapedType.getElementType();
  DenseElementsAttr denseAttr;
  if (mlir::isa<IntegerType>(elementType))
    denseAttr = DenseElementsAttr::get(shapedType, static_cast<int64_t>(val));
  else if (mlir::isa<FloatType>(elementType))
    denseAttr = DenseElementsAttr::get(shapedType, val);
  else
    llvm_unreachable("Unsupport type");
  MultiDialectBuilder<OnnxBuilder> create(builder, loc);
  return create.onnx.constant(denseAttr);
}

bool oneIsOfLayout(Type t1, Type t2,
    onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout layout) {
  if (auto rtp1 = llvm::dyn_cast<RankedTensorType>(t1)) {
    if (onnx_mlir::zhigh::getZTensorLayout(rtp1) == layout)
      return true;
    // t1 is not of `layout`, check t2.
    if (auto rtp2 = llvm::dyn_cast<RankedTensorType>(t2)) {
      return (onnx_mlir::zhigh::getZTensorLayout(rtp2) == layout);
    }
    // t2 is unranked.
  }
  // t1 is unranked.
  // Unranked type is potentially of `layout`.
  return true;
}

/// Check if ONNXReshapeOp is reshaping 2D to 4D by tiling each input dimension.
bool isTiling2DTo4D(Value val) {
  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  if (!reshapeOp)
    return false;

  Value input = reshapeOp.getData();
  Value output = reshapeOp.getReshaped();
  Type inputType = input.getType();
  Type outputType = output.getType();

  if (!isRankedShapedType(inputType))
    return false;
  if (!isRankedShapedType(outputType))
    return false;

  ArrayRef<int64_t> inputShape = getShape(inputType);
  ArrayRef<int64_t> outputShape = getShape(outputType);

  // Not reshape from 2D to 4D.
  if (!(inputShape.size() == 2 && outputShape.size() == 4))
    return false;

  // Tiling over each input dimension. Assume here that the dims are static.
  return ((inputShape[0] == outputShape[0] * outputShape[1]) &&
          (inputShape[1] == outputShape[2] * outputShape[3]));
}

/// Check if ONNXReshapeOp is reshaping 3D to 4D by tiling the first input
/// dimension.
bool isLeftmostTiling3DTo4D(Value val) {
  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  if (!reshapeOp)
    return false;

  Value input = reshapeOp.getData();
  Value output = reshapeOp.getReshaped();
  Type inputType = input.getType();
  Type outputType = output.getType();

  if (!isRankedShapedType(inputType))
    return false;
  if (!isRankedShapedType(outputType))
    return false;

  ArrayRef<int64_t> inputShape = getShape(inputType);
  ArrayRef<int64_t> outputShape = getShape(outputType);

  // Not reshape from 3D to 4D.
  if (!(inputShape.size() == 3 && outputShape.size() == 4))
    return false;

  // Tiling over each input dimension. Assume here that the dims are static.
  return ((inputShape[0] == outputShape[0] * outputShape[1]) &&
          (inputShape[1] == outputShape[2]) &&
          (inputShape[2] == outputShape[3]));
}

/// Check if ONNXReshapeOp is reshaping 3D to 4D by tiling the last input
/// dimension. If tilingSize>0, then check that it is tiling by that amount (or
/// a multiple thereof).
bool isRightmostTiling3DTo4D(Value val, int64_t tilingSize) {
  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  if (!reshapeOp)
    return false;

  Value input = reshapeOp.getData();
  Value output = reshapeOp.getReshaped();
  Type inputType = input.getType();
  Type outputType = output.getType();

  if (!isRankedShapedType(inputType))
    return false;
  if (!isRankedShapedType(outputType))
    return false;

  ArrayRef<int64_t> inputShape = getShape(inputType);
  ArrayRef<int64_t> outputShape = getShape(outputType);

  // Not reshape from 3D to 4D.
  if (!(inputShape.size() == 3 && outputShape.size() == 4))
    return false;

  // Check that the tiling size is given, then the last dim of the output is
  // statically determined and is a multiples of tiling size.
  if (tilingSize > 0)
    if (ShapedType::isDynamic(outputShape[3]) ||
        (outputShape[3] % tilingSize != 0))
      return false;

  // Tiling over each input dimension. Assume here that the dims are static.
  return ((inputShape[0] == outputShape[0]) &&
          (inputShape[1] == outputShape[1]) &&
          (inputShape[2] == outputShape[2] * outputShape[3]));
}

/// Check if a 4D tensor is collapsed into 2D by merging the each two
/// dimensions.
bool isCollapsing4DTo2D(Value val) {
  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  if (!reshapeOp)
    return false;

  Value input = reshapeOp.getData();
  Value output = reshapeOp.getReshaped();
  Type inputType = input.getType();
  Type outputType = output.getType();

  if (!isRankedShapedType(inputType))
    return false;
  if (!isRankedShapedType(outputType))
    return false;

  ArrayRef<int64_t> inputShape = getShape(inputType);
  ArrayRef<int64_t> outputShape = getShape(outputType);

  // Not reshape from 4D to 2D.
  if (!(inputShape.size() == 4 && outputShape.size() == 2))
    return false;

  // Collapsing by merging the first two dimensions. Assume here that the dims
  // are static.
  return ((inputShape[0] * inputShape[1] == outputShape[0]) &&
          (inputShape[2] * inputShape[3] == outputShape[1]));
}

/// Check if a 4D tensor is collapsed into 3D by merging the first two
/// (leftmost) dimensions.
bool isLeftmostCollapsing4DTo3D(Value val) {
  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  if (!reshapeOp)
    return false;

  Value input = reshapeOp.getData();
  Value output = reshapeOp.getReshaped();
  Type inputType = input.getType();
  Type outputType = output.getType();

  if (!isRankedShapedType(inputType))
    return false;
  if (!isRankedShapedType(outputType))
    return false;

  ArrayRef<int64_t> inputShape = getShape(inputType);
  ArrayRef<int64_t> outputShape = getShape(outputType);

  // Not reshape from 4D to 3D.
  if (!(inputShape.size() == 4 && outputShape.size() == 3))
    return false;

  // Collapsing by merging the first two dimensions. Assume here that the dims
  // are static.
  return ((inputShape[0] * inputShape[1] == outputShape[0]) &&
          (inputShape[2] == outputShape[1]) &&
          (inputShape[3] == outputShape[2]));
}

/// Check if a 4D tensor is collapsed into 3D by merging the last two
/// (rightmost) dimensions.
bool isRightmostCollapsing4DTo3D(Value val) {
  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  if (!reshapeOp)
    return false;

  Value input = reshapeOp.getData();
  Value output = reshapeOp.getReshaped();
  Type inputType = input.getType();
  Type outputType = output.getType();

  if (!isRankedShapedType(inputType))
    return false;
  if (!isRankedShapedType(outputType))
    return false;

  ArrayRef<int64_t> inputShape = getShape(inputType);
  ArrayRef<int64_t> outputShape = getShape(outputType);

  // Not reshape from 4D to 3D.
  if (!(inputShape.size() == 4 && outputShape.size() == 3))
    return false;

  // Collapsing by merging the first two dimensions. Assume here that the dims
  // are static.
  return ((inputShape[0] == outputShape[0]) &&
          (inputShape[1] == outputShape[1]) &&
          (inputShape[2] * inputShape[3] == outputShape[2]));
}

AffineMapAttr getTiling2DTo4DMap(OpBuilder &b, Value val) {
  assert(isTiling2DTo4D(val) &&
         "ONNXReshapeOp is not suitable for getting a tiling affine map");

  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  Value output = reshapeOp.getReshaped();
  Type outputType = output.getType();
  ArrayRef<int64_t> outputShape = getShape(outputType);

  AffineExpr tileConst1 = getAffineConstantExpr(outputShape[1], b.getContext());
  AffineExpr tileConst3 = getAffineConstantExpr(outputShape[3], b.getContext());

  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr d1 = b.getAffineDimExpr(1);

  AffineExpr o0 = d0.floorDiv(tileConst1);
  AffineExpr o1 = d0 % tileConst1;
  AffineExpr o2 = d1.floorDiv(tileConst3);
  AffineExpr o3 = d1 % tileConst3;

  AffineMap map = AffineMap::get(
      /*dims=*/2, /*symbols=*/0, {o0, o1, o2, o3}, b.getContext());
  return AffineMapAttr::get(map);
}

AffineMapAttr getLeftmostTiling3DTo4DMap(OpBuilder &b, Value val) {
  assert(isLeftmostTiling3DTo4D(val) &&
         "ONNXReshapeOp is not suitable for getting a tiling affine map");

  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  Value output = reshapeOp.getReshaped();
  Type outputType = output.getType();
  ArrayRef<int64_t> outputShape = getShape(outputType);

  AffineExpr tileConst1 = getAffineConstantExpr(outputShape[1], b.getContext());

  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr d1 = b.getAffineDimExpr(1);
  AffineExpr d2 = b.getAffineDimExpr(2);

  AffineExpr o0 = d0.floorDiv(tileConst1);
  AffineExpr o1 = d0 % tileConst1;

  AffineMap map = AffineMap::get(
      /*dims=*/3, /*symbols=*/0, {o0, o1, d1, d2}, b.getContext());
  return AffineMapAttr::get(map);
}

AffineMapAttr getCollapsing4DTo2DMap(OpBuilder &b, Value val) {
  assert(isCollapsing4DTo2D(val) &&
         "ONNXReshapeOp is not suitable for getting a collapsing affine map");

  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  Value input = reshapeOp.getData();
  Type inputType = input.getType();
  ArrayRef<int64_t> inputShape = getShape(inputType);

  AffineExpr dimConst1 = getAffineConstantExpr(inputShape[1], b.getContext());
  AffineExpr dimConst3 = getAffineConstantExpr(inputShape[3], b.getContext());

  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr d1 = b.getAffineDimExpr(1);
  AffineExpr d2 = b.getAffineDimExpr(2);
  AffineExpr d3 = b.getAffineDimExpr(3);

  AffineExpr o0 = d0 * dimConst1 + d1;
  AffineExpr o1 = d2 * dimConst3 + d3;

  AffineMap map = AffineMap::get(
      /*dims=*/4, /*symbols=*/0, {o0, o1}, b.getContext());
  return AffineMapAttr::get(map);
}

AffineMapAttr getLeftmostCollapsing4DTo3DMap(OpBuilder &b, Value val) {
  assert(isLeftmostCollapsing4DTo3D(val) &&
         "ONNXReshapeOp is not suitable for getting a collapsing affine map");

  auto reshapeOp = mlir::dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
  Value input = reshapeOp.getData();
  Type inputType = input.getType();
  ArrayRef<int64_t> inputShape = getShape(inputType);

  AffineExpr dimConst1 = getAffineConstantExpr(inputShape[1], b.getContext());

  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr d1 = b.getAffineDimExpr(1);
  AffineExpr d2 = b.getAffineDimExpr(2);
  AffineExpr d3 = b.getAffineDimExpr(3);

  AffineExpr o0 = d0 * dimConst1 + d1;

  AffineMap map = AffineMap::get(
      /*dims=*/4, /*symbols=*/0, {o0, d2, d3}, b.getContext());
  return AffineMapAttr::get(map);
}

AffineMapAttr getTransposeMap(OpBuilder &b, ArrayAttr permAttr) {
  SmallVector<unsigned, 4> perm;
  for (uint64_t i = 0; i < ArrayAttrSize(permAttr); ++i) {
    unsigned axis = ArrayAttrIntVal(permAttr, i);
    perm.emplace_back(axis);
  }
  AffineMap map =
      AffineMap::getPermutationMap(llvm::ArrayRef(perm), b.getContext());
  return AffineMapAttr::get(map);
}

/// Check the values of a transpose map to be equal to the permVals.
bool isTransposePermutationEqualTo(
    ArrayAttr permAttr, mlir::ArrayRef<int64_t> permVals) {
  // Check same rank.
  int64_t permAttrSize = ArrayAttrSize(permAttr);
  int64_t permValSize = permVals.size();
  if (permAttrSize != permValSize)
    return false;
  // Check same values; abort on failure.
  for (int64_t i = 0; i < permAttrSize; ++i) {
    int64_t v = ArrayAttrIntVal(permAttr, i);
    if (permVals[i] != v)
      return false;
  }
  // Identical, success.
  return true;
}

bool isShapeDimMultipleOf(Value val, int64_t index, int64_t multipleVal) {
  // Type must be shaped and ranked.
  Type type = val.getType();
  if (!isRankedShapedType(type))
    return false;
  // Index must be within bounds of the shape rank; negative is from back.
  ArrayRef<int64_t> shape = getShape(type);
  int64_t size = shape.size();
  if (index < 0)
    index += size;
  if (index < 0 || index >= size)
    return false;
  // At this time, only reason about static shapes.
  int64_t dim = shape[index];
  if (ShapedType::isDynamic(dim))
    return false;
  // All good now, check if dim is a multiple of "multipleVal."
  return dim % multipleVal == 0;
}

IntegerAttr getAxisNHWC(IntegerAttr axisNCHWAttr) {
  int64_t axisNCHW = axisNCHWAttr.getSInt();
  int64_t axisNHWC;
  switch (axisNCHW) {
  case 0: // N
    axisNHWC = 0;
    break;
  case 1: // C
    axisNHWC = 3;
    break;
  case 2: // H
    axisNHWC = 1;
    break;
  case 3: // W
    axisNHWC = 2;
    break;
  default:
    axisNHWC = axisNCHW;
  }
  return IntegerAttr::get(axisNCHWAttr.getType(), axisNHWC);
}

//===----------------------------------------------------------------------===//

bool hasNNPAUse(Value v) {
  return llvm::any_of(v.getUsers(), [](Operation *op) {
    // Stick/Unstick ops are not considered as NNPA ops.
    return ((op->getDialect()->getNamespace() ==
                ZHighDialect::getDialectNamespace()) &&
            !isa<ZHighStickOp, ZHighUnstickOp, ZHighStickForLSTMOp,
                ZHighStickForGRUOp>(op));
  });
}

/// Get default saturation setting.
IntegerAttr getDefaultSaturation(PatternRewriter &rewriter) {
  Type si64Ty = rewriter.getIntegerType(64, true);
  if (nnpaEnableSaturation)
    return rewriter.getIntegerAttr(si64Ty, -1);
  else
    return IntegerAttr();
}

} // namespace zhigh
} // namespace onnx_mlir
