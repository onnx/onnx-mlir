/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- LayoutHelper.cpp - NNPA Layout Helper ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include <numeric>

#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

zdnn_data_layouts convertLayoutAttrToZDNNDataLayout(
    int64_t rank, StringAttr layoutAttr) {
  zdnn_data_layouts zDNNDataLayout;
  if (layoutAttr) {
    llvm::StringRef layoutStr = layoutAttr.getValue();
    if (layoutStr.equals_insensitive(LAYOUT_1D))
      zDNNDataLayout = ZDNN_1D; // ZDNN_1D
    else if (layoutStr.equals_insensitive(LAYOUT_2D))
      zDNNDataLayout = ZDNN_2D; // ZDNN_2D
    else if (layoutStr.equals_insensitive(LAYOUT_2DS))
      zDNNDataLayout = ZDNN_2DS; // ZDNN_2DS
    else if (layoutStr.equals_insensitive(LAYOUT_3D))
      zDNNDataLayout = ZDNN_3D; // ZDNN_3D
    else if (layoutStr.equals_insensitive(LAYOUT_3DS))
      zDNNDataLayout = ZDNN_3DS; // ZDNN_3DS
    else if (layoutStr.equals_insensitive(LAYOUT_4D))
      zDNNDataLayout = ZDNN_4D; // ZDNN_4D
    else if (layoutStr.equals_insensitive(LAYOUT_4DS)) {
      zDNNDataLayout = ZDNN_4DS; // ZDNN_4DS
    } else if (layoutStr.equals_insensitive(LAYOUT_NHWC))
      zDNNDataLayout = ZDNN_NHWC; // ZDNN_NHWC
    else if (layoutStr.equals_insensitive(LAYOUT_NCHW))
      zDNNDataLayout = ZDNN_NCHW; // ZDNN_NCHW
    else if (layoutStr.equals_insensitive(LAYOUT_HWCK))
      zDNNDataLayout = ZDNN_HWCK; // ZDNN_CNNK_HWCK
    else if (layoutStr.equals_insensitive(LAYOUT_FICO))
      zDNNDataLayout = ZDNN_FICO; // ZDNN_FICO
    else if (layoutStr.equals_insensitive(LAYOUT_ZRH))
      zDNNDataLayout = ZDNN_ZRH; // ZDNN_ZRH
    else if (layoutStr.equals_insensitive(LAYOUT_BFICO))
      zDNNDataLayout = ZDNN_BIDIR_FICO; // ZDNN_BIDIR_FICO
    else if (layoutStr.equals_insensitive(LAYOUT_BZRH))
      zDNNDataLayout = ZDNN_BIDIR_ZRH; // ZDNN_BIDIR_ZRH
    else
      llvm_unreachable("Unsupported data layout");
  } else {
    // If zDNN data layout is not specified, use ZDNN_xD for a rank 'x'.
    if (rank == 1)
      zDNNDataLayout = ZDNN_1D; // ZDNN_1D
    else if (rank == 2)
      zDNNDataLayout = ZDNN_2D; // ZDNN_2D
    else if (rank == 3)
      // Use 3DS instead of 3D since important ops like LSTM/MatMul/Softmax use
      // 3DS, which reduces the number of layout transformations.
      zDNNDataLayout = ZDNN_3DS; // ZDNN_3DS
    else if (rank == 4)
      zDNNDataLayout = ZDNN_4D; // ZDNN_4D
    else
      llvm_unreachable("Unsupported data layout");
  }
  return zDNNDataLayout;
}

bool is2DLayout(StringAttr layout) {
  return (layout && layout.getValue().equals_insensitive(LAYOUT_2D));
}

bool is3DSLayout(StringAttr layout) {
  return (layout && layout.getValue().equals_insensitive(LAYOUT_3DS));
}

bool is4DLayout(StringAttr layout) {
  return (layout && layout.getValue().equals_insensitive(LAYOUT_4D));
}

bool is4DSLayout(StringAttr layout) {
  return (layout && layout.getValue().equals_insensitive(LAYOUT_4DS));
}

bool isNHWCLayout(StringAttr layout) {
  return (layout && layout.getValue().equals_insensitive(LAYOUT_NHWC));
}

mlir::StringAttr getNCHWLayoutAttr(PatternRewriter &rewriter) {
  return rewriter.getStringAttr(LAYOUT_NCHW);
}

SmallVector<int64_t, 4> convertTo4DShape(
    ArrayRef<int64_t> origShape, std::string layout) {
  SmallVector<int64_t, 4> shape4D;
  if (layout == LAYOUT_1D) {
    // (e1) -> (1, 1, 1, e1)
    assert(origShape.size() == 1 && "Shape and layout are inconsistent");
    shape4D.emplace_back(1);
    shape4D.emplace_back(1);
    shape4D.emplace_back(1);
    shape4D.emplace_back(origShape[0]);
  } else if (layout == LAYOUT_2D) {
    // (e2, e1) -> (1, 1, e2, e1)
    assert(origShape.size() == 2 && "Shape and layout are inconsistent");
    shape4D.emplace_back(1);
    shape4D.emplace_back(1);
    shape4D.emplace_back(origShape[0]);
    shape4D.emplace_back(origShape[1]);
  } else if (layout == LAYOUT_2DS) {
    // (e4, e1) -> (e4, 1, 1, e1)
    assert(origShape.size() == 2 && "Shape and layout are inconsistent");
    shape4D.emplace_back(origShape[0]);
    shape4D.emplace_back(1);
    shape4D.emplace_back(1);
    shape4D.emplace_back(origShape[1]);
  } else if (layout == LAYOUT_3D) {
    // (e3, e2, e1) -> (1, e3, e2, e1)
    assert(origShape.size() == 3 && "Shape and layout are inconsistent");
    shape4D.emplace_back(1);
    shape4D.emplace_back(origShape[0]);
    shape4D.emplace_back(origShape[1]);
    shape4D.emplace_back(origShape[2]);
  } else if (layout == LAYOUT_3DS) {
    // (e4, e2, e1) -> (e4, 1, e2, e1)
    assert(origShape.size() == 3 && "Shape and layout are inconsistent");
    shape4D.emplace_back(origShape[0]);
    shape4D.emplace_back(1);
    shape4D.emplace_back(origShape[1]);
    shape4D.emplace_back(origShape[2]);
  } else if (layout == LAYOUT_4D || layout == LAYOUT_4DS) {
    // (e4, e3, e2, e1) -> (e4, e3, e2, e1)
    assert(origShape.size() == 4 && "Shape and layout are inconsistent");
    for (int64_t v : origShape)
      shape4D.emplace_back(v);
  } else {
    llvm_unreachable("Unsupported data layout");
  }
  return shape4D;
}

bool isNoopReshape(ShapedType srcType, std::string srcLayout,
    ShapedType tgtType, std::string tgtLayout) {
  // Supported layouts: 2DS, 3DS, 4D.
  SmallVector<std::string> supportedLayouts = {
      LAYOUT_2DS, LAYOUT_3DS, LAYOUT_4D};
  if (llvm::none_of(supportedLayouts,
          [&srcLayout](std::string v) { return srcLayout == v; }))
    return false;
  if (llvm::none_of(supportedLayouts,
          [&tgtLayout](std::string v) { return tgtLayout == v; }))
    return false;

  // Both ztensors have static shape.
  if (!srcType.hasStaticShape() || !tgtType.hasStaticShape())
    return false;

  // Normalize to 4D shape.
  SmallVector<int64_t, 4> src4D =
      convertTo4DShape(srcType.getShape(), srcLayout);
  SmallVector<int64_t, 4> tgt4D =
      convertTo4DShape(tgtType.getShape(), tgtLayout);

  // Check total sizes.
  if (std::accumulate(src4D.begin(), src4D.end(), 1, std::multiplies<>{}) !=
      std::accumulate(tgt4D.begin(), tgt4D.end(), 1, std::multiplies<>{}))
    return false;

  // Memory access:
  // - (e4, e3, e2, e1) -> (e4, e1/64, e3, e2/32, e2%32, e1%64).
  //
  // Check that the shape's changed from
  // - S1(e4, e3, e2, e1)
  // to
  // - S2(e4', e3, e2, e1'),
  //
  // where
  // - e2 = e3 = 1
  // - e1 % 64 = 0 and e1' % 64 = 0
  //
  // which makes sure that values are at the same offset.
  if ((src4D[1] != tgt4D[1]) || (src4D[1] != 1))
    return false;
  if ((src4D[2] != tgt4D[2]) || (src4D[2] != 1))
    return false;
  if ((src4D[3] % 64 != 0) || (tgt4D[3] % 64 != 0))
    return false;

  return true;
}

} // namespace onnx_mlir
