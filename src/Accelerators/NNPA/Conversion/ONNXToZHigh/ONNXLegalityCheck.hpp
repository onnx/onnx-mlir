/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ONNXLegalityCheck.hpp - Check legality for ONNX ops -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods to check whether an ONNX op is dynamically
// legal or not. If an op is legal, it will NOT be lowered to ZHigh for ZAIU
// execution.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

/// A function to check whether an ONNX op is suitable for being lowered to zDNN
/// or not.
template <typename OP_TYPE>
bool isSuitableForZDNN(
    OP_TYPE op, const onnx_mlir::DimAnalysis *dimAnalysis = nullptr);

/// Get padding type using shape helper. This returns
/// `SAME_PADDING`, `VALID_PADDING`, or empty.
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
mlir::StringRef getStrPaddingType(OP op);

/// Check if input, output, kernel, strides, and paddingYype for each axis meet
/// parameter restrictions for maxpool and avgpool.
/// See "MaxPool2D/AvgPool2D Parameter Restrictions" in "zDNN API Reference"
bool meetPoolParamRestrictions(int64_t inputShape, int64_t kernelShape,
    int64_t strides, int64_t outputShape, mlir::StringRef paddingType);
