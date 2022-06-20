
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

void populateAffineAndKrnlToLLVMConversion(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx,
    llvm::ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps,
    bool verifyInputTensors);

void populateKrnlToLLVMConversion(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx,
    llvm::ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps,
    bool verifyInputTensors);

void populateLoweringKrnlCallOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlEntryPointOpPattern(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx,
    llvm::ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps,
    bool verifyInputTensors);

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
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void determineOwnershipForOutputOMTensors(mlir::ModuleOp &module,
    llvm::SmallVectorImpl<bool> &outputOMTensorOwnerships);

void recordEntryPointSignatures(mlir::ModuleOp &module,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps);

void genSignatureFunction(mlir::ModuleOp &module,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    const llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps);
} // namespace krnl
} // namespace onnx_mlir
