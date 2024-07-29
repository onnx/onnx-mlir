/*
 * SPDX-License-Identifier: Apache-2.0
 */
//====------ ConvertSeqToMemrefM.hpp - Krnl Dialect Lowering
//---------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Lowering of Krnl operations to a combination of other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_CONVERT_SEQ_H
#define ONNX_MLIR_CONVERT_SEQ_H

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

namespace onnx_mlir {
namespace krnl {

void populateLoweringKrnlSeqAllocOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlSeqDeallocOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlSeqExtractOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlSeqStoreOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

} // namespace krnl
} // namespace onnx_mlir
#endif
