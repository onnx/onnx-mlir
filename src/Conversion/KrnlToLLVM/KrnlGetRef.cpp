/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlGetRefOp.cpp - Lower KrnlGetRefOp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlGetRefOp operator.
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

class KrnlGetRefOpLowering : public ConvertToLLVMPattern {
public:
  using ConvertToLLVMPattern::createIndexAttrConstant;
  using ConvertToLLVMPattern::getIndexType;

  explicit KrnlGetRefOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConvertToLLVMPattern(
            KrnlGetRefOp::getOperationName(), context, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    KrnlGetRefOpAdaptor operandAdaptor(operands);

    // This is the type of the krnl.getref output. This type is used
    // for the type of the internal MemRef.
    auto type = op->getResult(0).getType();
    auto memRefTy = type.cast<mlir::MemRefType>();

    // auto llvmMemRefType = typeConverter->convertType(type).cast<Type>();
    auto outputElementType =
        typeConverter->convertType(memRefTy.getElementType());

    // This is the start of the memory pool containing the output MemRef.
    Type memPoolType = operandAdaptor.getMempool()
                           .getType()
                           .cast<LLVM::LLVMStructType>()
                           .getBody()[1];
    Value alignedMemPoolBase =
        create.llvm.extractValue(memPoolType, operandAdaptor.getMempool(), {1});

    // Get pointer using the offset.
    auto llvmOutputElementType = outputElementType.cast<Type>();
    auto offset = operandAdaptor.getOffset();
    auto llvmMemPoolType = typeConverter->convertType(memPoolType).cast<Type>();
    auto outputMemPoolTypePtrAlloc =
        create.llvm.getElemPtr(llvmMemPoolType, llvmOutputElementType,
            alignedMemPoolBase, ArrayRef<LLVM::GEPArg>{offset});

    // Bitcast to output MemRef type i.e. from i8* to the element type
    // of the output MemRef.
    Value outputTypedPtrAlloc =
        create.llvm.bitcast(getPointerType(context, llvmOutputElementType),
            outputMemPoolTypePtrAlloc);

    // Handle the static case.
    if (hasAllConstantDimensions(memRefTy)) {
      // Create llvm MemRef from original MemRef and fill the data pointers.
      auto llvmMemRef = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memRefTy, outputTypedPtrAlloc);

      rewriter.replaceOp(op, {llvmMemRef});
      return success();
    }

    // Handle the dynamic case.

    // Compute strides and offset based on MemRef type.
    int64_t alignmentOffset;
    SmallVector<int64_t, 4> strides;
    auto successStrides =
        getStridesAndOffset(memRefTy, strides, alignmentOffset);
    (void)successStrides;
    assert(succeeded(successStrides) && "unexpected non-strided memref");

    // Create the memRef descriptor.
    auto structType = typeConverter->convertType(memRefTy);
    auto memRefDescriptor = MemRefDescriptor::undef(rewriter, loc, structType);

    // Allocated pointer, used for malloc/free.
    memRefDescriptor.setAllocatedPtr(rewriter, loc, outputTypedPtrAlloc);

    // Actual aligned pointer to payload.
    // TODO: support aligned MemRefs.
    memRefDescriptor.setAlignedPtr(rewriter, loc, outputTypedPtrAlloc);

    // Offset in aligned pointer.
    // TODO: support non-zero here in the aligned case.

    Type indexType = getIndexType();
    memRefDescriptor.setOffset(
        rewriter, loc, createIndexAttrConstant(rewriter, loc, indexType, 0));

    if (memRefTy.getRank() != 0) {
      // Prepare sizes.
      SmallVector<Value, 4> sizes;
      sizes.reserve(memRefTy.getRank());
      unsigned i = 0;
      for (int64_t s : memRefTy.getShape())
        sizes.push_back(
            s == ShapedType::kDynamic
                ? operands[2 + i++]
                : createIndexAttrConstant(rewriter, loc, indexType, s));

      // Store all sizes in the descriptor. Only dynamic sizes are passed in as
      // operands to AllocOp.
      Value runningStride = nullptr;
      auto nStrides = strides.size();
      SmallVector<Value, 4> strideValues(nStrides, nullptr);
      for (unsigned i = 0; i < nStrides; ++i) {
        int64_t index = nStrides - 1 - i;
        if (strides[index] == ShapedType::kDynamic)
          // Identity layout map is enforced in the match function, so we
          // compute:
          //   `runningStride *= sizes[index + 1]`
          runningStride = runningStride ? rewriter.create<LLVM::MulOp>(loc,
                                              runningStride, sizes[index + 1])
                                        : createIndexAttrConstant(
                                              rewriter, loc, indexType, 1);
        else
          runningStride =
              createIndexAttrConstant(rewriter, loc, indexType, strides[index]);
        strideValues[index] = runningStride;
      }
      // Fill size and stride descriptors in memref.
      for (auto indexedSize : llvm::enumerate(sizes)) {
        int64_t index = indexedSize.index();
        memRefDescriptor.setSize(rewriter, loc, index, indexedSize.value());
        memRefDescriptor.setStride(rewriter, loc, index, strideValues[index]);
      }
    }

    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }
};

void populateLoweringKrnlGetRefOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlGetRefOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
