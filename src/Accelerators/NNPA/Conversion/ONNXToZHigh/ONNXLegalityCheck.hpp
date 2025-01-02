/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ONNXLegalityCheck.hpp - Check legality for ONNX ops -------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods to check whether an ONNX op is dynamically
// legal or not. If an op is legal, it will NOT be lowered to ZHigh for ZAIU
// execution.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_LEGALITY_H
#define ONNX_MLIR_LEGALITY_H

#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

/// A function to check whether a value's element type is valid for zAIU or not.
/// zAIU supports only F16, F32 and BFLOAT. Since MLIR does not support BFLOAT,
/// we check F16 and F32 here only. zAIU only supports rank in range of (0, 4].
bool isValidElementTypeAndRank(
    mlir::Operation *op, mlir::Value val, bool donotCheckRank = false);

/// A function to check whether an ONNX op is suitable for being lowered to zDNN
/// or not.
template <typename OP_TYPE>
bool isSuitableForZDNN(
    OP_TYPE op, const onnx_mlir::DimAnalysis *dimAnalysis = nullptr);

/// Check if the input NNPA level is compatible with the current NNPA
/// level.
bool isCompatibleWithNNPALevel(std::string inputNNPALevel);

/// Get padding type using shape helper. This returns
/// `SAME_PADDING`, `VALID_PADDING`, or empty.
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
mlir::StringRef getStrPaddingType(OP op);

/// Check if input, output, kernel, strides, and paddingYype for each axis meet
/// parameter restrictions for maxpool and avgpool.
/// See "MaxPool2D/AvgPool2D Parameter Restrictions" in "zDNN API Reference"
bool meetPoolParamRestrictions(mlir::Operation *op, int64_t inputShape,
    int64_t kernelShape, int64_t strides, int64_t outputShape,
    mlir::StringRef paddingType);

bool onnxToZHighUnsupportedReport(
    mlir::Operation *op, const std::string &message);

bool onnxToZHighInCompatibilityReport(
    mlir::Operation *op, const std::string &message);

bool onnxToZHighInCompatibilityReport(mlir::Operation *op, NNPALevel level);

#endif
