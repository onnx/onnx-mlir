/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlNone.cpp - Lower KrnlNoneOp -------------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlNoneOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlNoneOpLowering : public ConversionPattern {
public:
  explicit KrnlNoneOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlNoneOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    LLVMBuilder create(rewriter, loc);
    MLIRContext *context = rewriter.getContext();
    Value nullPtr = create.null(getI8PointerType(context));
    rewriter.replaceOp(op, nullPtr);
    return success();
  }
};

void populateLoweringKrnlNoneOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlNoneOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
