/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToLLVM.hpp - Krnl Dialect Lowering  ---------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Lowering of Krnl operations to a combination of other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_CONVERT_KRNL_TO_LLVM_H
#define ONNX_MLIR_CONVERT_KRNL_TO_LLVM_H

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
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &inputMemRefTypes,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &outputMemRefTypes,
    bool verifyInputTensors, bool enableParallel);

void populateKrnlToLLVMConversion(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx,
    llvm::ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &inputMemRefTypes,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &outputMemRefTypes,
    bool verifyInputTensors);

void populateLoweringKrnlCallOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlEntryPointOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx, llvm::ArrayRef<bool> constantOutputs,
    bool singleEntryPoint,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &entryGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &inSigGlobalOps,
    llvm::SmallVectorImpl<mlir::LLVM::GlobalOp> &outSigGlobalOps,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &inputMemRefTypes,
    std::map<std::string, llvm::SmallVector<mlir::MemRefType, 4>>
        &outputMemRefTypes,
    bool verifyInputTensors);

void populateLoweringKrnlFindIndexOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlGlobalOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlInstrumentOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlMemcpyOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlPrintOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlPrintTensorOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlRandomNormalOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlStrlenOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlStrncmpOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlUnaryMathOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlVectorTypeCastOpPattern(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

void populateLoweringKrnlNoneOpPattern(mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringKrnlRoundEvenOpPattern(
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
#endif
