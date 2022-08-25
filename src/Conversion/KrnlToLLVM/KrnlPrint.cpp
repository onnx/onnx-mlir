/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlPrint.cpp - Lower KrnlPrintOp -----------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlPrintOp operator.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlPrintOpLowering : public ConversionPattern {
public:
  explicit KrnlPrintOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlPrintOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto printOp = cast<KrnlPrintOp>(op);
    Location loc = printOp.getLoc();
    KrnlPrintOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    Value input = operandAdaptor.input();
    StringRef format = printOp.format();
    ModuleOp module = printOp->getParentOfType<ModuleOp>();

    // Get a symbol reference to the runtime function to use, creating one if
    // necessary.
    auto printfFuncRef = getOrInsertPrintf(rewriter, module);

    // Printf call.
    LLVM::GlobalOp formatSpec = getOrCreateGlobalString(format, loc, rewriter,
        module, static_cast<LLVMTypeConverter *>(getTypeConverter()));
    Value formatSpecPtr = getPtrToGlobalString(formatSpec, loc, rewriter);

    if (input)
      create.llvm.call({}, printfFuncRef, {formatSpecPtr, input});
    else
      create.llvm.call({}, printfFuncRef, {formatSpecPtr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  static FlatSymbolRefAttr getOrInsertPrintf(
      PatternRewriter &rewriter, ModuleOp module) {
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    MLIRContext *ctx = rewriter.getContext();
    Type voidType = LLVM::LLVMVoidType::get(ctx);
    Type i8Type = IntegerType::get(ctx, 8);
    Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
    return create.llvm.getOrInsertSymbolRef(module, StringRef("printf"),
        voidType, {i8PtrType},
        /*isVarArg=*/true);
  }
};

void populateLoweringKrnlPrintOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlPrintOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
