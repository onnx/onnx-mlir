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
    auto memsetOp = cast<KrnlMemsetOp>(op);
    bool delayed = memsetOp.delayed();
    KrnlMemsetOpAdaptor operandAdaptor(memsetOp);
    Value destMemRef(operandAdaptor.dest());
    Value destVal(operandAdaptor.value());
    Location loc = memsetOp.getLoc();

    // If delayed but the input memref has not normalized yet, do nothing.
    if (delayed &&
        !destMemRef.getType().cast<MemRefType>().getLayout().isIdentity())
      return failure();

    AffineBuilderKrnlMem createAffine(rewriter, loc);
    IndexExprScope indexScope(createAffine);
    MemRefBoundsIndexCapture destBounds(destMemRef);

    int rank = destBounds.getRank();
    SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> ubs;
    destBounds.getDimList(ubs);
    SmallVector<int64_t, 4> steps(rank, 1);
    // Copy data,
    createAffine.forIE(lbs, ubs, steps,
        [&](AffineBuilderKrnlMem &createAffine, ValueRange indices) {
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
