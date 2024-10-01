/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- KrnlGetLinearOffsetIndex.cpp - -----------------------===//
//
// Copyright 2024- The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlGetLinearOffsetIndexOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlGetLinearOffsetIndexLowering : public ConversionPattern {
public:
  explicit KrnlGetLinearOffsetIndexLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter,
            KrnlGetLinearOffsetIndexOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MultiDialectBuilder<IndexExprBuilderForKrnl> create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    auto krnlOp = llvm::cast<KrnlGetLinearOffsetIndexOp>(op);
    KrnlGetLinearOffsetIndexOpAdaptor operandAdaptor(krnlOp);
    // Get the input memref.
    Value memref = operandAdaptor.getMemref();
    // Get indices.
    SmallVector<Value, 4> mapOperands(krnlOp.getMapOperands());
    auto mapResults = mlir::affine::expandAffineMap(
        rewriter, loc, krnlOp.getMap(), mapOperands);
    if (!mapResults)
      return failure();
    SmallVector<Value, 8> indices = mapResults.value();

    auto memrefTy = llvm::dyn_cast<MemRefType>(memref.getType());
    int64_t rank = memrefTy.getRank();
    assert(static_cast<int64_t>(mapResults.value().size()) == rank &&
           "Invalid indices");

    // Only lower this op after the memref is normalized.
    if (!memrefTy.getLayout().isIdentity())
      return failure();

    // Get dimension sizes.
    SmallVector<IndexExpr, 4> dims;
    create.krnlIE.getShapeAsDims(memref, dims);
    // Compute the linear offset using strides.
    IndexExpr offsetIE = LitIE(0);
    IndexExpr strideIE = LitIE(1);
    for (int64_t i = rank - 1; i >= 0; --i) {
      IndexExpr strideOffset = strideIE * DimIE(indices[i]);
      offsetIE = offsetIE + strideOffset;
      if (i > 0)
        strideIE = strideIE * dims[i];
    }

    rewriter.replaceOp(op, offsetIE.getValue());
    return success();
  }
};

void populateLoweringKrnlGetLinearOffsetIndexOpPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    MLIRContext *ctx) {
  patterns.insert<KrnlGetLinearOffsetIndexLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
