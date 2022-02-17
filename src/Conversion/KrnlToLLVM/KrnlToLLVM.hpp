/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlToLLVM.hpp - Lowering from KRNL+Affine+Std to LLVM -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#ifndef KRNL_TO_LLVM_H
#define KRNL_TO_LLVM_H

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"

namespace mlir {

class MLIRContext;
class LLVMTypeConverter;

class RewritePatternSet;

void checkConstantOutputs(
    ModuleOp &module, SmallVectorImpl<bool> &constantOutputs);

void recordEntryPointSignatures(ModuleOp &module,
    SmallVectorImpl<std::string> &entryPointNames,
    SmallVectorImpl<std::string> &inSignatures,
    SmallVectorImpl<std::string> &outSignatures);

void genSignatureFunction(ModuleOp module,
    ArrayRef<std::string> entryPointNames, ArrayRef<std::string> inSignatures,
    ArrayRef<std::string> outSignatures);

void populateAffineAndKrnlToLLVMConversion(RewritePatternSet &patterns,
    MLIRContext *ctx, LLVMTypeConverter &typeConverter,
    ArrayRef<bool> constantOutputs, bool singleEntryPoint);

} // namespace mlir

#endif // KRNL_TO_LLVM_H
