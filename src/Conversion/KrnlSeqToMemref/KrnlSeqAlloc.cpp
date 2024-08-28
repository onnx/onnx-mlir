/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlSeqAlloc.cpp - Lower KrnlSeqAllocOp ------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlSeqAllocOp operator.
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

class KrnlSeqAllocOpLowering : public ConversionPattern {
public:
  explicit KrnlSeqAllocOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlSeqAllocOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlSeqAllocOpAdaptor operandAdaptor(operands);
    KrnlSeqAllocOp thisOp = mlir::dyn_cast<KrnlSeqAllocOp>(op);
    Location loc = op->getLoc();
    MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);

    Value outputSeq = thisOp.getResult();
    auto outputType = mlir::cast<MemRefType>(outputSeq.getType());
    Value alloc;
    if (outputType.isDynamicDim(0)) {
      llvm::SmallVector<Value, 4> length(operandAdaptor.getLength());
      alloc = create.mem.alloc(outputType, length);
    } else
      alloc = create.mem.alloc(outputType);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringKrnlSeqAllocOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlSeqAllocOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
