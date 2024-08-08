/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==--- QuantizeHelper.hpp - Helper functions for Quantization Op lowering --=//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains definitions of helper functions for quantization lowering.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

namespace onnx_mlir {

// Given an input, scale, zero point, qMin, and qMax, perform a linear
// quantization and store in alloc.
void emitQuantizationLinearScalarParameters(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Operation *op, mlir::MemRefType inputType,
    mlir::MemRefType quantizedType, mlir::Value alloc, DimsExpr &allocDims,
    mlir::Value input, mlir::Value qMin, mlir::Value qMax, mlir::Value scale,
    mlir::Value zeroPoint, bool enableSIMD, bool enableParallel);

// Scan the input to compute scale, zeroPoint, and quantizedZeroPoint given qMin
// and qMax.
void emitDynamicQuantizationLinearScalarParameters(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Operation *op, mlir::MemRefType inputType,
    mlir::MemRefType quantizedType, mlir::Value input, mlir::Value qMin,
    mlir::Value qMax, mlir::Value &scale, mlir::Value &zeroPoint,
    mlir::Value &quantizedZeroPoint, bool enableSIMD, bool enableParallel);
} // namespace onnx_mlir
