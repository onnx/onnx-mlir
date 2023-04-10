/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlSeqDealloc.cpp - Lower KrnlSeqDeallocOp ------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlSeqDeallocOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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

class KrnlSeqDeallocOpLowering : public ConversionPattern {
public:
  explicit KrnlSeqDeallocOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlSeqDeallocOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlSeqDeallocOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);

    auto input_sequence = operandAdaptor.getInputSequence();
    auto dimSize = create.mem.dim(input_sequence, 0);
    rewriter.create<scf::ForOp>(loc, create.math.constantIndex(0), dimSize,
        create.math.constantIndex(1), ValueRange(),
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value forInduction,
            ValueRange iterArgs) {
          MultiDialectBuilder<MathBuilder, MemRefBuilder> create(
              bodyBuilder, bodyLoc);
          auto element = bodyBuilder.create<memref::LoadOp>(
              bodyLoc, operandAdaptor.getInputSequence(), forInduction);
          create.mem.dealloc(element);
          bodyBuilder.create<scf::YieldOp>(bodyLoc);
        });

    create.mem.dealloc(input_sequence);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateLoweringKrnlSeqDeallocOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlSeqDeallocOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
