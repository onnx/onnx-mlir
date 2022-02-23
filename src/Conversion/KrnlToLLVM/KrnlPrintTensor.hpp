/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlPrintTensor.hpp - Lower KrnlPrintTensorOp -----------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the lowering class for the KrnlPrintTensorOp operator.
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

class KrnlPrintTensorOpLowering : public ConversionPattern {
public:
  explicit KrnlPrintTensorOpLowering(
      MLIRContext *context, TypeConverter &typeConverter)
      : ConversionPattern(
            typeConverter, KrnlPrintTensorOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

} // namespace onnx_mlir
