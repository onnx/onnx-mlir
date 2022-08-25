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
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto memcpyRef = getOrInsertMemcpy(rewriter, parentModule);

    // First operand.
    Type dstType = operandAdaptor.dest()
                       .getType()
                       .cast<LLVM::LLVMStructType>()
                       .getBody()[1];
    Value alignedDstMemory =
        create.llvm.extractValue(dstType, operandAdaptor.dest(), {1});
    Value alignedInt8PtrDstMemory = create.llvm.bitcastI8Ptr(alignedDstMemory);

    // Second operand.
    Type srcType = operandAdaptor.src()
                       .getType()
                       .cast<LLVM::LLVMStructType>()
                       .getBody()[1];
    Value alignedSrcMemory =
        create.llvm.extractValue(srcType, operandAdaptor.src(), {1});
    Value alignedInt8PtrSrcMemory = create.llvm.bitcastI8Ptr(alignedSrcMemory);

    // Size.
    Value int64Size = rewriter.create<LLVM::SExtOp>(
        loc, IntegerType::get(context, 64), operandAdaptor.size());

    // Is volatile (set to false).
    Value isVolatile =
        create.llvm.constant(IntegerType::get(context, 1), (int64_t)0);

    // Memcpy call
    create.llvm.call({}, memcpyRef,
        {alignedInt8PtrDstMemory, alignedInt8PtrSrcMemory, int64Size,
            isVolatile});

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Return a symbol reference to the memcpy function, inserting it into the
  /// module if necessary.
  FlatSymbolRefAttr getOrInsertMemcpy(
      PatternRewriter &rewriter, ModuleOp module) const {
    MLIRContext *context = module.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    // Create a function declaration for memcpy, the signature is:
    //   * `void (i8*, i8* , i64, i1)`
    Type llvmVoidTy = LLVM::LLVMVoidType::get(context);
    Type llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    Type llvmI64Ty = IntegerType::get(context, 64);
    Type llvmI1Ty = IntegerType::get(context, 1);
    return create.llvm.getOrInsertSymbolRef(module,
        StringRef("llvm.memcpy.p0.p0.i64"), llvmVoidTy,
        {llvmI8PtrTy, llvmI8PtrTy, llvmI64Ty, llvmI1Ty});
  }
};

void populateLoweringKrnlMemcpyOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlMemcpyOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
