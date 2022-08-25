/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlStrlend.cpp - Lower KrnlStrlenOp --------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlStrlenOp operator.
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

class KrnlStrlenOpLowering : public ConversionPattern {
public:
  explicit KrnlStrlenOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlStrlenOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    KrnlStrlenOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Get a symbol reference to the strlen function, inserting it if necessary.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr strlenRef = getOrInsertStrlen(rewriter, module);

    // Operand.
    MLIRContext *ctx = module.getContext();
    Type i8Type = IntegerType::get(ctx, 8);
    Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
    Value strPtr =
        rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrType, operandAdaptor.str());

    // Strlen call.
    // TODO: should return a size_t
    Type retType = IntegerType::get(context, 64);
    auto funcCall = rewriter.create<func::CallOp>(
        loc, strlenRef, retType, ArrayRef<Value>({strPtr}));

    rewriter.replaceOp(op, funcCall.getResults()[0]);
    return success();
  }

private:
  /// Return a symbol reference to the strlen function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertStrlen(
      PatternRewriter &rewriter, ModuleOp module) {
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    // Create 'strlen' function signature: `size_t (i8*)`
    // TODO: need to create size_t not i64.
    MLIRContext *ctx = module.getContext();
    Type i8Type = IntegerType::get(ctx, 8);
    Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
    return create.llvm.getOrInsertSymbolRef(
        module, StringRef("strlen"), rewriter.getI64Type(), {i8PtrType});
  }
};

void populateLoweringKrnlStrlenOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlStrlenOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
