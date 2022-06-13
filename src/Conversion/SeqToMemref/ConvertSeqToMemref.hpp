/*
 * SPDX-License-Identifier: Apache-2.0
 */
//====------ ConvertSeqToMemrefM.hpp - Krnl Dialect Lowering
//---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Lowering of Krnl operations to a combination of other dialects.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

namespace onnx_mlir {
namespace krnl {

void populateLoweringKrnlSeqExtractOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlSeqStoreOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

} // namespace krnl
} // namespace onnx_mlir
