/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlVectorTypeCastOp.cpp - Lower KrnlVectorTypeCastOp ---------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlVectorTypeCastOp operator.
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

class KrnlVectorTypeCastOpLowering : public ConvertToLLVMPattern {
public:
  explicit KrnlVectorTypeCastOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConvertToLLVMPattern(
            KrnlVectorTypeCastOp::getOperationName(), context, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto krnlVectorTypeCastOp = mlir::cast<KrnlVectorTypeCastOp>(op);
    MemRefType sourceType =
        mlir::cast<MemRefType>(krnlVectorTypeCastOp.getOperand().getType());
    MemRefType targetType = krnlVectorTypeCastOp.getType();
    if (!isSupportedMemRefType(targetType) ||
        !isSupportedMemRefType(sourceType))
      return failure();

    KrnlVectorTypeCastOp::Adaptor transformed(operands);
    MemRefDescriptor srcMemRefDesc(transformed.getSource());

    Type targetStructType =
        typeConverter->convertType(krnlVectorTypeCastOp.getType());
    if (!targetStructType)
      return failure();

    Location loc = op->getLoc();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    // Get memRefDescriptor, the new memref descriptor.
    MemRefDescriptor memRefDescriptor =
        MemRefDescriptor::undef(rewriter, loc, targetStructType);
    auto targetElementPtrType = memRefDescriptor.getElementPtrType();

    // Set the new memref to the same buffer as the source memref.
    Value srcBuffer = srcMemRefDesc.allocatedPtr(rewriter, loc);
    Value targetBuffer = create.llvm.bitcast(targetElementPtrType, srcBuffer);
    memRefDescriptor.setAllocatedPtr(rewriter, loc, targetBuffer);

    // Set the new memref alignment to the same value as source memref.
    Value srcBufferAligned = srcMemRefDesc.alignedPtr(rewriter, loc);
    Value targetBufAligned =
        create.llvm.bitcast(targetElementPtrType, srcBufferAligned);
    memRefDescriptor.setAlignedPtr(rewriter, loc, targetBufAligned);

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(getStridesAndOffset(targetType, strides, offset)))
      return failure();

    // Unhandled dynamic offset.
    if (offset == ShapedType::kDynamic)
      return failure();

    Type indexType = ConvertToLLVMPattern::getIndexType();

    memRefDescriptor.setOffset(rewriter, loc,
        createIndexAttrConstant(rewriter, loc, indexType, offset));

    // Get the sizes of the memref: all but the last one are copied from the
    // source memref. If the dimension size was static, the target memref would
    // have the same size.
    SmallVector<Value, 4> sizes;
    sizes.reserve(targetType.getRank());
    for (unsigned pos = 0, e = targetType.getRank() - 1; pos < e; ++pos) {
      int64_t dimSize = targetType.getDimSize(pos);
      if (ShapedType::isDynamic(dimSize))
        sizes.push_back(srcMemRefDesc.size(rewriter, loc, pos));
      else
        sizes.push_back(
            createIndexAttrConstant(rewriter, loc, indexType, dimSize));
    }

    if (!ShapedType::isDynamic(targetType.getShape().back())) {
      // The op is already verified to have the right size for the last
      // dimension.
      sizes.push_back(createIndexAttrConstant(
          rewriter, loc, indexType, targetType.getShape().back()));
    } else {
      // We need to divide the dynamic size on the source by the vector width.
      // There is the implicit expectation that the last dimension of the
      // original memory is a multiple of the vector length.
      Value vecWidth = createIndexAttrConstant(rewriter, loc, indexType,
          mlir::cast<ShapedType>(targetType.getElementType()).getNumElements());
      sizes.push_back(rewriter.create<LLVM::UDivOp>(loc,
          srcMemRefDesc.size(rewriter, loc, sourceType.getRank() - 1),
          vecWidth));
    }

    assert(!sizes.empty() && "target memref rank can't be zero");

    // Compute the total number of memref elements.
    Value cumulativeSize = sizes.front();
    for (unsigned i = 1, e = sizes.size(); i < e; ++i)
      cumulativeSize = rewriter.create<LLVM::MulOp>(
          loc, getIndexType(), ArrayRef<Value>{cumulativeSize, sizes[i]});

    // Calculate the strides.
    Value runningStride = nullptr;
    // Iterate strides in reverse order, compute runningStride and strideValues.
    unsigned nStrides = strides.size();
    SmallVector<Value, 4> strideValues(nStrides, nullptr);
    for (auto indexedStride : llvm::enumerate(llvm::reverse(strides))) {
      int64_t index = nStrides - 1 - indexedStride.index();
      if (strides[index] == ShapedType::kDynamic)
        // Identity layout map is enforced in the match function, so we compute:
        //   `runningStride *= sizes[index + 1]`.
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

    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }

  // Check if the MemRefType `type` is supported by the lowering. We currently
  // only support memrefs with identity maps.
  bool isSupportedMemRefType(MemRefType type) const {
    if (!typeConverter->convertType(type.getElementType()))
      return false;
    return type.getLayout().isIdentity();
  }
};

void populateLoweringKrnlVectorTypeCastOpPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    MLIRContext *ctx) {
  patterns.insert<KrnlVectorTypeCastOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
