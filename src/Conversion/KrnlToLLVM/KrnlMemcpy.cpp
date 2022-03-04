/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlMemcpy.cpp - Lower KrnlMemcpyOp ---------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlMemcpyOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlMemcpyOpLowering : public ConversionPattern {
public:
  explicit KrnlMemcpyOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlMemcpyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    KrnlMemcpyOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto memcpyRef = getOrInsertMemcpy(rewriter, parentModule);

    // First operand.
    Type dstType = operandAdaptor.dest()
                       .getType()
                       .cast<LLVM::LLVMStructType>()
                       .getBody()[1];
    Value alignedDstMemory = rewriter.create<LLVM::ExtractValueOp>(
        loc, dstType, operandAdaptor.dest(), rewriter.getI64ArrayAttr(1));
    Value alignedInt8PtrDstMemory = rewriter.create<LLVM::BitcastOp>(loc,
        LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
        alignedDstMemory);

    // Second operand.
    Type srcType = operandAdaptor.src()
                       .getType()
                       .cast<LLVM::LLVMStructType>()
                       .getBody()[1];
    Value alignedSrcMemory = rewriter.create<LLVM::ExtractValueOp>(
        loc, srcType, operandAdaptor.src(), rewriter.getI64ArrayAttr(1));
    Value alignedInt8PtrSrcMemory = rewriter.create<LLVM::BitcastOp>(loc,
        LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
        alignedSrcMemory);

    // Size.
    Value int64Size = rewriter.create<LLVM::SExtOp>(
        loc, IntegerType::get(context, 64), operandAdaptor.size());

    // Is volatile (set to false).
    Value isVolatile =
        rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(context, 1),
            rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));

    // Memcpy call
    rewriter.create<CallOp>(loc, memcpyRef, ArrayRef<Type>({}),
        ArrayRef<Value>({alignedInt8PtrDstMemory, alignedInt8PtrSrcMemory,
            int64Size, isVolatile}));

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Return a symbol reference to the memcpy function, inserting it into the
  /// module if necessary.
  FlatSymbolRefAttr getOrInsertMemcpy(
      PatternRewriter &rewriter, ModuleOp module) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.memcpy.p0i8.p0i8.i64"))
      return SymbolRefAttr::get(context, "llvm.memcpy.p0i8.p0i8.i64");
    // Create a function declaration for memcpy, the signature is:
    //   * `void (i8*, i8* , i64, i1)`
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmI1Ty = IntegerType::get(context, 1);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy,
        ArrayRef<mlir::Type>({llvmI8PtrTy, llvmI8PtrTy, llvmI64Ty, llvmI1Ty}),
        false);

    // Insert the memcpy function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "llvm.memcpy.p0i8.p0i8.i64", llvmFnType);
    return SymbolRefAttr::get(context, "llvm.memcpy.p0i8.p0i8.i64");
  }
};

void populateLoweringKrnlMemcpyOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlMemcpyOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
