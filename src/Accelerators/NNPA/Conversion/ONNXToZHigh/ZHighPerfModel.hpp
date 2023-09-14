/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZHighPerfModel.hpp - Deciding ONNX vs ZHigh for ops -------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains model info to help decide for the relevant NNPA ops if
// they are faster / slower than their equivalent CPU versions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

// Return true if operation is faster on NNPA than CPU.

// Result is normalized for add/sub/mul. Operations that have an additional
// advantage on the NNPA vs CPU execution can reflect that advantage via the
// relativeNNPASpeedup ratio.

// Elementwise with one input operand.
bool isElementwiseFasterOnNNPA(mlir::Operation *op, mlir::Value operand,
    const onnx_mlir::DimAnalysis *dimAnalysis,
    double relativeNNPASpeedup = 1.0);

// Elementwise with two input operands, lhs and rhs.
bool isElementwiseFasterOnNNPA(mlir::Operation *op, mlir::Value lhs,
    mlir::Value rhs, const onnx_mlir::DimAnalysis *dimAnalysis,
    double relativeNNPASpeedup = 1.0);

bool isMatMulFasterOnNNPA(mlir::Operation *op, mlir::Value a, mlir::Value b,
    bool aTransposed, bool bTransposed,
    const onnx_mlir::DimAnalysis *dimAnalysis);
