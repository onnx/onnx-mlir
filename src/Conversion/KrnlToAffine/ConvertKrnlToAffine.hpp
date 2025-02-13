/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToAffine.hpp - Krnl Dialect Lowering --------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the lowering of Krnl operations to the affine dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_CONVERT_KRNL_TO_AFFINE_H
#define ONNX_MLIR_CONVERT_KRNL_TO_AFFINE_H

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

namespace onnx_mlir {
namespace krnl {

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//
class AffineTypeConverter : public mlir::TypeConverter {
public:
  using mlir::TypeConverter::TypeConverter;

  AffineTypeConverter();

  /// Return true if the inputs and outputs of the given function type are
  /// legal. [Taken from MLIR and adapted to only check the legality of the
  /// inputs. Once unranked results can be handled gracefully this
  /// override needs to be removed in favour of the original MLIR one.]
  bool isSignatureLegal(mlir::FunctionType funcType) {
    return llvm::all_of(llvm::concat<const mlir::Type>(
                            funcType.getInputs(), funcType.getResults()),
        [this](mlir::Type type) { return isLegal(type); });
  }

  /// Return true if the operands/results of call have a legal type.
  bool isSignatureLegal(mlir::func::CallOp call) {
    auto f = [this](mlir::Type type) { return isLegal(type); };
    return llvm::all_of(call.getOperandTypes(), f) &&
           llvm::all_of(call.getResultTypes(), f);
  }
};

// To assist unroll and jam
using UnrollAndJamRecord = std::pair<mlir::affine::AffineForOp, int64_t>;
using UnrollAndJamList = llvm::SmallVector<UnrollAndJamRecord, 4>;
using UnrollAndJamMap = std::map<mlir::Operation *, UnrollAndJamList *>;

void populateKrnlToAffineConversion(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx,
    bool enableParallel);

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

void populateLoweringKrnlGetLinearOffsetIndexOpPattern(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlMatmultOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx,
    bool parallelEnabled);

void populateLoweringKrnlMemsetOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlPrefetchOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlTerminatorOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

} // namespace krnl
} // namespace onnx_mlir
#endif
