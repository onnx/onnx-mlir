
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToLLVM.hpp - Krnl Dialect Lowering  ---------------===//
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

const std::string DEFAULT_DYN_ENTRY_POINT = "run_main_graph";

namespace onnx_mlir {
namespace krnl {

void populateKrnlToLLVMConversion(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx,
    llvm::ArrayRef<bool> constantOutputs, bool singleEntryPoint);

void populateLoweringKrnlEntryPointOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx,
    llvm::ArrayRef<bool> constantOutputs, bool singleEntryPoint);

void populateLoweringKrnlFindIndexOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlGetRefOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlGlobalOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlInstrumentOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlMemcpyOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlPrintOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlPrintTensorOpPattern(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlRandomNormalOpPattern(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlStrlenOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlStrncmpOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlUnaryMathOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlVectorTypeCastOpPattern(
    LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void recordEntryPointSignatures(mlir::ModuleOp &module,
    llvm::SmallVectorImpl<std::string> &entryPointNames,
    llvm::SmallVectorImpl<std::string> &inSignatures,
    llvm::SmallVectorImpl<std::string> &outSignatures);

void genSignatureFunction(mlir::ModuleOp module,
    const llvm::ArrayRef<std::string> entryPointNames,
    const llvm::ArrayRef<std::string> inSignatures,
    const llvm::ArrayRef<std::string> outSignatures);
} // namespace krnl
} // namespace onnx_mlir
