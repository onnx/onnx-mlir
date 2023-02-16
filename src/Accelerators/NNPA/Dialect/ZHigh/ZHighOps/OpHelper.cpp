/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- OpHelper.cpp - NNPA ZHigh Helper Functions ------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
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
  ShapedType shapedType = val.getType().cast<ShapedType>();
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
    return ZTensorEncodingAttr::DataLayout::_3D;
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

//===----------------------------------------------------------------------===//
// Utility functions to query ztensor information.

bool isZTensor(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    if (ttp.getEncoding().dyn_cast_or_null<ZTensorEncodingAttr>())
      return true;
  return false;
}

ZTensorEncodingAttr getZTensorEncoding(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<ZTensorEncodingAttr>();
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

//===----------------------------------------------------------------------===//
// Utility functions.

Value getMinusBcastConst(
    mlir::OpBuilder &builder, Location loc, FloatAttr floatAttr, Value X) {
  ShapedType xType = X.getType().cast<ShapedType>();
  assert(xType.hasStaticShape() && "expected static shape");
  float val = floatAttr.getValueAsDouble() * -1.0;
  DenseElementsAttr denseAttr = DenseElementsAttr::get(X.getType(), val);
  MultiDialectBuilder<OnnxBuilder> create(builder, loc);
  return create.onnx.constant(denseAttr);
}

bool oneIsOfNHWCLayout(Type t1, Type t2) {
  if (auto rtp1 = llvm::dyn_cast<RankedTensorType>(t1)) {
    if (onnx_mlir::zhigh::getZTensorLayout(rtp1) ==
        onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::NHWC)
      return true;
    // t1 is not of NHWC, check t2.
    if (auto rtp2 = llvm::dyn_cast<RankedTensorType>(t2)) {
      return (onnx_mlir::zhigh::getZTensorLayout(rtp2) ==
              onnx_mlir::zhigh::ZTensorEncodingAttr::DataLayout::NHWC);
    }
    // t2 is unranked.
  }
  // t1 is unranked.
  // Unranked type is potentially of NHWC.
  return true;
}

/// Check if ONNXReshapeOp is reshaping 2D to 4D by tiling each input dimension.
bool isTiling2DTo4D(Value val) {
  auto reshapeOp = dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
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

  // Tiling over each input dimension.
  return ((inputShape[0] == outputShape[0] * outputShape[1]) &&
          (inputShape[1] == outputShape[2] * outputShape[3]));
}

/// Check if ONNXReshapeOp is reshaping 3D to 4D by tiling the first input
/// dimension.
bool isTiling3DTo4D(Value val) {
  auto reshapeOp = dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
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

  // Tiling over each input dimension.
  return ((inputShape[0] == outputShape[0] * outputShape[1]) &&
          (inputShape[1] == outputShape[2]) &&
          (inputShape[2] == outputShape[3]));
}

/// Check if a 4D tensor is collapsed into 2D by merging the each two
/// dimensions.
bool isCollapsing4DTo2D(Value val) {
  auto reshapeOp = dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
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

  // Collapsing by merging the first two dimensions.
  return ((inputShape[0] * inputShape[1] == outputShape[0]) &&
          (inputShape[2] * inputShape[3] == outputShape[1]));
}

/// Check if a 4D tensor is collapsed into 3D by merging the first two
/// dimensions.
bool isCollapsing4DTo3D(Value val) {
  auto reshapeOp = dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
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

  // Collapsing by merging the first two dimensions.
  return ((inputShape[0] * inputShape[1] == outputShape[0]) &&
          (inputShape[2] == outputShape[1]) &&
          (inputShape[3] == outputShape[2]));
}

AffineMapAttr getTiling2DTo4DMap(OpBuilder &b, Value val) {
  assert(isTiling2DTo4D(val) &&
         "ONNXReshapeOp is not suitable for getting a tiling affine map");

  auto reshapeOp = dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
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

AffineMapAttr getTiling3DTo4DMap(OpBuilder &b, Value val) {
  assert(isTiling3DTo4D(val) &&
         "ONNXReshapeOp is not suitable for getting a tiling affine map");

  auto reshapeOp = dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
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

  auto reshapeOp = dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
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

AffineMapAttr getCollapsing4DTo3DMap(OpBuilder &b, Value val) {
  assert(isCollapsing4DTo3D(val) &&
         "ONNXReshapeOp is not suitable for getting a collapsing affine map");

  auto reshapeOp = dyn_cast<ONNXReshapeOp>(val.getDefiningOp());
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

} // namespace zhigh
} // namespace onnx_mlir
