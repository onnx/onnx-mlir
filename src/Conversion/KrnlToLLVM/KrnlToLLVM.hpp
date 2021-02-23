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
class OwningRewritePatternList;

void populateAffineAndKrnlToLLVMConversion(OwningRewritePatternList &patterns,
    MLIRContext *ctx, LLVMTypeConverter &typeConverter);

} // namespace mlir

#endif // KRNL_TO_LLVM_H
