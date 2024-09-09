/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlMemcpy.cpp - Lower KrnlMemcpyOp ---------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
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
#include "src/Support/KrnlSupport.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlMemcpyOpLowering : public ConversionPattern {
public:
  explicit KrnlMemcpyOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlMemcpyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    KrnlMemcpyOp memcpyOp = llvm::dyn_cast<KrnlMemcpyOp>(op);

    // Get operands.
    KrnlMemcpyOpAdaptor operandAdaptor(operands);
    Value src = operandAdaptor.getSrc();
    Value dest = operandAdaptor.getDest();
    Value srcOffset = operandAdaptor.getSrcOffset();
    Value dstOffset = operandAdaptor.getDestOffset();
    Value elemsToCopy = operandAdaptor.getNumElems();

    // Common types.
    Type i1Ty = IntegerType::get(context, 1);
    Type i64Ty = IntegerType::get(context, 64);
    Type i8PtrTy = getPointerType(context, IntegerType::get(context, 8));
    Type elementType =
        mlir::cast<LLVM::LLVMStructType>(src.getType()).getBody()[1];
    int64_t eltSize = getMemRefEltSizeInBytes(
        mlir::dyn_cast<MemRefType>(memcpyOp.getSrc().getType()));
    Value eltSizeInBytes = create.llvm.constant(i64Ty, eltSize);

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto memcpyRef = getOrInsertMemcpy(rewriter, parentModule);

    // First operand.
    Value alignedDstMemory = create.llvm.extractValue(elementType, dest, {1});
    // Update the pointer with the given offset.
    Value dstPtrInInt = create.llvm.ptrtoint(i64Ty, alignedDstMemory);
    Value dstOffsetI64 = create.llvm.bitcast(i64Ty, dstOffset);
    Value dstOffsetInBytes = create.llvm.mul(dstOffsetI64, eltSizeInBytes);
    dstPtrInInt = create.llvm.add(dstPtrInInt, dstOffsetInBytes);
    alignedDstMemory = create.llvm.inttoptr(elementType, dstPtrInInt);
    Value dstAddress = create.llvm.bitcast(i8PtrTy, alignedDstMemory);

    // Second operand.
    Value alignedSrcMemory = create.llvm.extractValue(elementType, src, {1});
    // Update the pointer with the given offset.
    Value srcPtrInInt = create.llvm.ptrtoint(i64Ty, alignedSrcMemory);
    Value srcOffsetI64 = create.llvm.bitcast(i64Ty, srcOffset);
    Value srcOffsetInBytes = create.llvm.mul(srcOffsetI64, eltSizeInBytes);
    srcPtrInInt = create.llvm.add(srcPtrInInt, srcOffsetInBytes);
    alignedSrcMemory = create.llvm.inttoptr(elementType, srcPtrInInt);
    Value srcAddress = create.llvm.bitcast(i8PtrTy, alignedSrcMemory);

    // Size.
    Value sizeInBytes = create.llvm.mul(elemsToCopy, eltSizeInBytes);

    // Is volatile (set to false).
    Value isVolatile = create.llvm.constant(i1Ty, static_cast<int64_t>(0));

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
    Type llvmI8PtrTy = getPointerType(context, IntegerType::get(context, 8));
    Type llvmI64Ty = IntegerType::get(context, 64);
    Type llvmI1Ty = IntegerType::get(context, 1);
    return create.llvm.getOrInsertSymbolRef(module,
        StringRef("llvm.memcpy.p0.p0.i64"), llvmVoidTy,
        {llvmI8PtrTy, llvmI8PtrTy, llvmI64Ty, llvmI1Ty});
  }
};

void populateLoweringKrnlMemcpyOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlMemcpyOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
