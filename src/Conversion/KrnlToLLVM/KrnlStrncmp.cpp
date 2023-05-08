/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlStrncmp.cpp - Lower KrnlStrncmpOp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlStrncmpOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlStrncmpOpLowering : public ConversionPattern {
public:
  explicit KrnlStrncmpOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlStrncmpOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlStrncmpOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Get a symbol reference to the strncmp function, inserting it if
    // necessary.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr StrncmpRef = getOrInsertStrncmp(rewriter, module);

    // Operands.
    MLIRContext *ctx = module.getContext();
    Type i8Type = IntegerType::get(ctx, 8);
    Type i8PtrType = getPointerType(ctx, i8Type);
    Value str1Ptr = rewriter.create<LLVM::IntToPtrOp>(
        loc, i8PtrType, operandAdaptor.getStr1());
    Value str2Ptr = rewriter.create<LLVM::IntToPtrOp>(
        loc, i8PtrType, operandAdaptor.getStr2());
    Value length = operandAdaptor.getLen();

    // Strncmp call.
    Type i32Type = IntegerType::get(ctx, 32);
    auto funcCall = rewriter.create<func::CallOp>(
        loc, StrncmpRef, i32Type, ArrayRef<Value>({str1Ptr, str2Ptr, length}));

    rewriter.replaceOp(op, funcCall.getResults()[0]);
    return success();
  }
};

void populateLoweringKrnlStrncmpOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlStrncmpOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
