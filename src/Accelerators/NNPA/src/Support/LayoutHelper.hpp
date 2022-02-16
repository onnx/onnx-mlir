//===---------- LayoutHelper.cpp - DLC++ Layout Helper --------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/zdnn-lib/zdnn.h"

/// Note: Keep these strings in sync with the one in Dialect/ZHigh/ZHighOps.td.
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

zdnn_data_layouts convertLayoutAttrToZDNNDataLayout(
    int64_t rank, mlir::StringAttr layoutAttr);
