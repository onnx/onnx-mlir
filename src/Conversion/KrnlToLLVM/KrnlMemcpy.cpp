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
#include "src/Dialect/Mlir/DialectBuilder.hpp"
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
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    // Get operands.
    KrnlMemcpyOpAdaptor operandAdaptor(operands);
    Value src = operandAdaptor.src();
    Value dest = operandAdaptor.dest();
    Value srcOffset = operandAdaptor.src_offset();
    Value dstOffset = operandAdaptor.dest_offset();
    Value copySize = operandAdaptor.size();

    // Common types.
    Type i1Ty = IntegerType::get(context, 1);
    Type i64Ty = IntegerType::get(context, 64);
    Type srcType = src.getType().cast<LLVM::LLVMStructType>().getBody()[1];
    Type dstType = dest.getType().cast<LLVM::LLVMStructType>().getBody()[1];

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto memcpyRef = getOrInsertMemcpy(rewriter, parentModule);

    // First operand.
    Value alignedDstMemory = create.llvm.extractValue(dstType, dest, {1});
    // Update the pointer with the given offset.
    Value dstPtrInInt = create.llvm.ptrtoint(i64Ty, alignedDstMemory);
    dstPtrInInt = create.llvm.add(dstPtrInInt, dstOffset);
    alignedDstMemory = create.llvm.inttoptr(dstType, dstPtrInInt);
    Value dstAddress = create.llvm.bitcastI8Ptr(alignedDstMemory);

    // Second operand.
    Value alignedSrcMemory = create.llvm.extractValue(srcType, src, {1});
    // Update the pointer with the given offset.
    Value srcPtrInInt = create.llvm.ptrtoint(i64Ty, alignedSrcMemory);
    srcPtrInInt = create.llvm.add(srcPtrInInt, srcOffset);
    alignedSrcMemory = create.llvm.inttoptr(srcType, srcPtrInInt);
    Value srcAddress = create.llvm.bitcastI8Ptr(alignedSrcMemory);

    // Size.
    Value sizeInBytes = create.llvm.sext(i64Ty, copySize);

    // Is volatile (set to false).
    Value isVolatile = create.llvm.constant(i1Ty, (int64_t)0);

    // Memcpy call
    create.llvm.call(
        {}, memcpyRef, {dstAddress, srcAddress, sizeInBytes, isVolatile});

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
