/*
 * SPDX-License-Identifier: Apache-2.0
 */
//===------ KrnlSeqStore.cpp - Lower KrnlSeqStoreOp ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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
    auto loc = op->getLoc();
    MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);

    // Allocate a new tensor and copy input tensor into it
    auto inputType = operandAdaptor.input().getType().cast<MemRefType>();
    SmallVector<mlir::Value, 4> allocParams;
    for (size_t i = 0; i < inputType.getShape().size(); i++) {
      if (inputType.getShape()[i] == -1) {
        allocParams.emplace_back(create.mem.dim(operandAdaptor.input(), i));
      }
    }
    Value alloc = create.mem.alignedAlloc(inputType, allocParams);
    rewriter.create<memref::CopyOp>(loc, operandAdaptor.input(), alloc);

    // Cast the input tensor to the element type of the sequence
    auto seq = operandAdaptor.seq();
    auto seqElementType =
        seq.getType().cast<MemRefType>().getElementType().cast<MemRefType>();
    auto casted = create.mem.cast(alloc, seqElementType);

    // Store the tensor
    rewriter.create<memref::StoreOp>(loc, casted, seq, operandAdaptor.index());

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
