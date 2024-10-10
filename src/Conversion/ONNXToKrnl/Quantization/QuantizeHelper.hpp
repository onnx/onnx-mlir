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
// quantization and store in alloc. FastMath enables taking the reciprocal for
// faster results on machines where mul is faster than div.
void emitQuantizationLinearScalarParameters(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Operation *op, mlir::MemRefType inputType,
    mlir::MemRefType quantizedType, mlir::Value alloc, DimsExpr &allocDims,
    mlir::Value input, mlir::Value qMin, mlir::Value qMax, mlir::Value scale,
    mlir::Value zeroPoint, bool hasZeroPoint, bool enableSIMD,
    bool enableParallel, bool enableFastMath);

// Scan the input to compute scale, zeroPoint, and quantizedZeroPoint given qMin
// and qMax.
void emitDynamicQuantizationLinearScalarParameters(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Operation *op, mlir::MemRefType inputType,
    mlir::MemRefType quantizedType, mlir::Value input, mlir::Value qMin,
    mlir::Value qMax, mlir::Value &scale, mlir::Value &zeroPoint,
    mlir::Value &quantizedZeroPoint, bool wantZeroPoint, bool enableSIMD,
    bool enableParallel);
} // namespace onnx_mlir
