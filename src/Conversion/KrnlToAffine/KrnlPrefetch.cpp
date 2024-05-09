/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- KrnlGetLinearOffsetIndex.cpp - -----------------------===//
//
// Copyright 2024- The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlPrefetchOp operator.
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

class KrnlPrefetchOpLowering : public ConversionPattern {
public:
  explicit KrnlPrefetchOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlPrefetchOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MultiDialectBuilder<AffineBuilder> create(rewriter, loc);

    auto krnlOp = llvm::cast<KrnlPrefetchOp>(op);
    KrnlPrefetchOpAdaptor operandAdaptor(krnlOp);

    Operation *affineOp = create.affine.prefetch(operandAdaptor.getMemref(),
        operandAdaptor.getMap(), operandAdaptor.getIndices(),
        operandAdaptor.getIsWrite(), operandAdaptor.getLocalityHint(),
        operandAdaptor.getIsDataCache());

    rewriter.replaceOp(op, affineOp);
    return success();
  }
};

void populateLoweringKrnlPrefetchOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlPrefetchOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
