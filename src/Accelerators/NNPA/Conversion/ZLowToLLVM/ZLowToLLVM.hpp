/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZLowToLLVM.hpp - Lowering from ZLow to LLVM ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace onnx_mlir {
namespace zlow {

/// Populate all conversion patterns for ZLow Ops.
void populateZLowToLLVMConversionPattern(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx);

} // namespace zlow
} // namespace onnx_mlir
