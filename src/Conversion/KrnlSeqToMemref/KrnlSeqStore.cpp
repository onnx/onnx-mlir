/*
 * SPDX-License-Identifier: Apache-2.0
 */
//===------ KrnlSeqStore.cpp - Lower KrnlSeqStoreOp ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlSeqStoreOp operator.
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
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlSeqStoreOpLowering : public ConversionPattern {
public:
  explicit KrnlSeqStoreOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlSeqStoreOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlSeqStoreOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);

    // Allocate a new tensor and copy input tensor into it
    auto inputType =
        mlir::cast<MemRefType>(operandAdaptor.getInput().getType());
    SmallVector<Value, 4> allocParams;
    for (size_t i = 0; i < inputType.getShape().size(); i++) {
      if (inputType.isDynamicDim(i)) {
        allocParams.emplace_back(create.mem.dim(operandAdaptor.getInput(), i));
      }
    }
    Value alloc = create.mem.alignedAlloc(inputType, allocParams);
    rewriter.create<memref::CopyOp>(loc, operandAdaptor.getInput(), alloc);

    // Cast the input tensor to the element type of the sequence
    auto seq = operandAdaptor.getSeq();
    auto seqElementType = mlir::cast<MemRefType>(
        mlir::cast<MemRefType>(seq.getType()).getElementType());
    auto casted = create.mem.cast(alloc, seqElementType);

    // Store the tensor
    rewriter.create<memref::StoreOp>(
        loc, casted, seq, operandAdaptor.getIndex());

    rewriter.eraseOp(op);
    return success();
  }
};

void populateLoweringKrnlSeqStoreOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlSeqStoreOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
