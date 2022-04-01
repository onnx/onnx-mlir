/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToAffine.hpp - Krnl Dialect Lowering --------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the lowering of Krnl operations to the affine dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

namespace onnx_mlir {
namespace krnl {

// To assist unroll and jam
using UnrollAndJamRecord = std::pair<mlir::AffineForOp, int64_t>;
using UnrollAndJamList = llvm::SmallVector<UnrollAndJamRecord, 4>;
using UnrollAndJamMap = std::map<mlir::Operation *, UnrollAndJamList *>;

void populateKrnlToAffineConversion(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlCopyFromBufferOpPattern(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlCopyToBufferOpPattern(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlLoadOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlStoreOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlMatmultOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlMemsetOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlTerminatorOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

} // namespace krnl
} // namespace onnx_mlir
