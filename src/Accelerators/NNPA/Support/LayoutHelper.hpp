/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- LayoutHelper.hpp - NNPA Layout Helper ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_LAYOUT_HELPER_H
#define ONNX_MLIR_LAYOUT_HELPER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "zdnn.h"

namespace onnx_mlir {

/// Note: Keep these strings in sync with the one in Dialect/ZHigh/ZHigh.td.
const std::string LAYOUT_1D = "1D";
const std::string LAYOUT_2D = "2D";
const std::string LAYOUT_3D = "3D";
const std::string LAYOUT_4D = "4D";
const std::string LAYOUT_2DS = "2DS";
const std::string LAYOUT_3DS = "3DS";
const std::string LAYOUT_4DS = "4DS";
const std::string LAYOUT_NHWC = "NHWC";
const std::string LAYOUT_NCHW = "NCHW";
const std::string LAYOUT_HWCK = "HWCK";
const std::string LAYOUT_FICO = "FICO";
const std::string LAYOUT_ZRH = "ZRH";
const std::string LAYOUT_BFICO = "BFICO";
const std::string LAYOUT_BZRH = "BZRH";

// Quantized transform type.
const std::string QTYPE_DLFLOAT16 = "DLFLOAT16";
const std::string QTYPE_INT8 = "INT8";
const std::string QTYPE_WEIGHTS = "WEIGHTS";
const std::string QTYPE_UNDEFINED = "UNDEFINED";

zdnn_data_layouts convertLayoutAttrToZDNNDataLayout(
    int64_t rank, mlir::StringAttr layoutAttr);

bool is2DLayout(mlir::StringAttr layout);
bool is3DSLayout(mlir::StringAttr layout);
bool is4DLayout(mlir::StringAttr layout);
bool is4DSLayout(mlir::StringAttr layout);
bool isNHWCLayout(mlir::StringAttr layout);

mlir::StringAttr getNCHWLayoutAttr(mlir::PatternRewriter &rewriter);

// A function to normalize a shape of a specific layout to a shape of 4D layout.
// This is useful because the 4D layout is used for stickification.
mlir::SmallVector<int64_t, 4> convertTo4DShape(
    mlir::ArrayRef<int64_t> origShape, std::string layout);

// A function to check if reshaping a ztensor to a new ztensor of different
// shape is "no-op" or not.
// "no-op" means that logically elements in the reshaped ztensor are in the same
// order and offset as the ones in the original ztensor.
bool isNoopReshape(mlir::ShapedType srcType, std::string srcLayout,
    mlir::ShapedType tgtType, std::string tgtLayout);

} // namespace onnx_mlir
#endif
