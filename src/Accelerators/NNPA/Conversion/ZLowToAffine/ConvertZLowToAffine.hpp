/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertZLowToAffine.hpp - ZLow Dialect Lowering --------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the lowering of ZLow operations to the affine dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Transforms/DialectConversion.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

namespace onnx_mlir {
namespace zlow {

void populateZLowToAffineConversion(mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

void populateLoweringZLowConvertDLF16OpPattern(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext *ctx);

} // namespace zlow
} // namespace onnx_mlir
