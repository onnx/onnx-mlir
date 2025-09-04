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

bool isNoopReshape(ShapedType srcType, std::string srcLayout,
    ShapedType tgtType, std::string tgtLayout) {
  if (srcLayout != tgtLayout)
    return false;

  // Supported layout: 3DS, 4D.
  if (llvm::none_of(SmallVector<std::string>{LAYOUT_2DS, LAYOUT_3DS, LAYOUT_4D},
          [&srcLayout](std::string v) { return srcLayout == v; }))
    return false;

  // Both ztensors have static shape.
  if (!srcType.hasStaticShape() || !tgtType.hasStaticShape())
    return false;

  // Normalize to 4D shape.
  SmallVector<int64_t, 4> srcShape, tgtShape;
  ArrayRef<int64_t> S1 = srcType.getShape();
  ArrayRef<int64_t> S2 = tgtType.getShape();
  if (srcLayout == LAYOUT_2DS) {
    // source.
    srcShape.emplace_back(S1[0]);
    srcShape.emplace_back(1);
    srcShape.emplace_back(1);
    srcShape.emplace_back(S1[1]);
    // target.
    tgtShape.emplace_back(S2[0]);
    tgtShape.emplace_back(1);
    tgtShape.emplace_back(1);
    tgtShape.emplace_back(S2[1]);
  } else if (srcLayout == LAYOUT_3DS) {
    // source.
    srcShape.emplace_back(S1[0]);
    srcShape.emplace_back(1);
    srcShape.emplace_back(S1[1]);
    srcShape.emplace_back(S1[2]);
    // target.
    tgtShape.emplace_back(S2[0]);
    tgtShape.emplace_back(1);
    tgtShape.emplace_back(S2[1]);
    tgtShape.emplace_back(S2[2]);
  } else if (srcLayout == LAYOUT_4D) {
    for (uint64_t i = 0; i < S1.size(); ++i) {
      srcShape.emplace_back(S1[i]);
      tgtShape.emplace_back(S2[i]);
    }
  } else {
    return false;
  }

  // Check total sizes.
  if (std::accumulate(
          srcShape.begin(), srcShape.end(), 1, std::multiplies<>{}) !=
      std::accumulate(tgtShape.begin(), tgtShape.end(), 1, std::multiplies<>{}))
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
  if ((srcShape[1] != tgtShape[1]) || (srcShape[1] != 1))
    return false;
  if ((srcShape[2] != tgtShape[2]) || (srcShape[2] != 1))
    return false;
  if ((srcShape[3] % 64 != 0) || (tgtShape[3] % 64 != 0))
    return false;

  return true;
}

} // namespace onnx_mlir
