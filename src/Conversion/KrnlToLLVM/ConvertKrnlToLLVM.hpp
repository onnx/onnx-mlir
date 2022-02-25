
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

#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

const std::string DEFAULT_DYN_ENTRY_POINT = "run_main_graph";

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

void populateKrnlToLLVMConversion(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> constantOutputs, bool singleEntryPoint);

void populateLoweringKrnlEntryPointOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> constantOutputs, bool singleEntryPoint);

void populateLoweringKrnlFindIndexOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlGetRefOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlGlobalOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlInstrumentOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlMemcpyOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlPrintOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlPrintTensorOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlRandomNormalOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlStrlenOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlStrncmpOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlUnaryMathOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx);

void populateLoweringKrnlVectorTypeCastOpPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    MLIRContext *ctx);

void recordEntryPointSignatures(ModuleOp &module,
    SmallVectorImpl<std::string> &entryPointNames,
    SmallVectorImpl<std::string> &inSignatures,
    SmallVectorImpl<std::string> &outSignatures);

void genSignatureFunction(ModuleOp module,
    const ArrayRef<std::string> entryPointNames,
    const ArrayRef<std::string> inSignatures,
    const ArrayRef<std::string> outSignatures);
} // namespace krnl
} // namespace onnx_mlir
