/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlPrint.hpp - Lower KrnlPrintOp -----------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the lowering class for the KrnlPrintOp operator.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Support/Common.hpp"

using namespace mlir;

namespace onnx_mlir {

class KrnlPrintOpLowering : public ConversionPattern {
public:
  explicit KrnlPrintOpLowering(
      MLIRContext *context, TypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter, KrnlPrintOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;

private:
  static FlatSymbolRefAttr getOrInsertPrintf(
      PatternRewriter &rewriter, ModuleOp module);
};

} // namespace onnx_mlir
