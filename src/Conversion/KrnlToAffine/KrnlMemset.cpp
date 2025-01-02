/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- KrnlMemset.cpp - Lower KrnlMemsetOp -------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlMemsetOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlMemsetLowering : public ConversionPattern {
public:
  explicit KrnlMemsetLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlMemsetOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Get info from operands.
    auto memsetOp = mlir::cast<KrnlMemsetOp>(op);
    bool delayed = memsetOp.getDelayed();
    KrnlMemsetOpAdaptor operandAdaptor(memsetOp);
    Value destMemRef(operandAdaptor.getDest());
    Value destVal(operandAdaptor.getValue());
    Location loc = memsetOp.getLoc();

    // If delayed but the input memref has not normalized yet, do nothing.
    if (delayed &&
        !mlir::cast<MemRefType>(destMemRef.getType()).getLayout().isIdentity())
      return failure();

    // TODO, Flatten, and possibly parallelize/simd. Maybe add a mode to detect
    // if/when mem override is allowed.
    MultiDialectBuilder<AffineBuilderKrnlMem, IndexExprBuilderForKrnl> create(
        rewriter, loc);
    IndexExprScope indexScope(create.affineKMem);
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(destMemRef, ubs);
    int rank = ubs.size();
    SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));
    SmallVector<int64_t, 4> steps(rank, 1);
    SmallVector<bool, 4> useParallel(rank, false);
    // Copy data,
    create.affineKMem.forLoopsIE(lbs, ubs, steps, useParallel,
        [&](const AffineBuilderKrnlMem &createAffine, ValueRange indices) {
          createAffine.store(destVal, destMemRef, indices);
        });
    rewriter.eraseOp(op);
    return success();
  }
};

void populateLoweringKrnlMemsetOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlMemsetLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
