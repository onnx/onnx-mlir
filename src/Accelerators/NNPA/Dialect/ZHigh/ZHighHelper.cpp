/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZHighHelper.cpp - NNPA ZHigh Helper Functions ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zhigh {

/// Check if a value type is ranked or unranked.
bool hasRankedType(Value val) {
  ShapedType shapedType = val.getType().cast<ShapedType>();
  return (shapedType && shapedType.hasRank());
}

/// Get a ztensor data layout by StringAttr.
ZTensorEncodingAttr::DataLayout convertStringAttrToDataLayout(
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
ZTensorEncodingAttr::DataLayout getDataLayoutByRank(int64_t rank) {
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
StringAttr convertDataLayoutToStringAttr(
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

Value getMinusBcastConst(
    mlir::OpBuilder &builder, Location loc, FloatAttr floatAttr, Value X) {
  ShapedType xType = X.getType().cast<ShapedType>();
  assert(xType.hasStaticShape() && "expected static shape");
  float val = floatAttr.getValueAsDouble() * -1.0;
  DenseElementsAttr denseAttr = DenseElementsAttr::get(X.getType(), val);
  MultiDialectBuilder<OnnxBuilder> create(builder, loc);
  return create.onnx.constant(denseAttr);
}

} // namespace zhigh
} // namespace onnx_mlir
