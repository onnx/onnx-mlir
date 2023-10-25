/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ZLowAttachLayout.cpp - Lower ZLowAttachLayoutOp ---------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ZLowAttachLayoutOp operator to affine dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

#include "src/Accelerators/NNPA/Conversion/ZLowToAffine/ConvertZLowToAffine.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "zlow_to_affine"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

class ZLowAttachLayoutLowering
    : public OpConversionPattern<ZLowAttachLayoutOp> {
public:
  using OpConversionPattern<ZLowAttachLayoutOp>::OpConversionPattern;
  using OpAdaptor = typename ZLowAttachLayoutOp::Adaptor;

  LogicalResult matchAndRewrite(ZLowAttachLayoutOp detachOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = detachOp.getLoc();
    Value inputMemref = adaptor.getInput();
    MemRefType outputType = detachOp.getOutput().getType().cast<MemRefType>();

    MultiDialectBuilder<AffineBuilder, MemRefBuilder, IndexExprBuilderForZLow>
        create(rewriter, loc);
    IndexExprScope indexScope(create.affine);

    SmallVector<IndexExpr, 4> ubs;
    create.zlowIE.getShapeAsDims(inputMemref, ubs);
    int rank = ubs.size();

    // Allocate the output buffer.
    Value alloc = create.mem.alignedAlloc(outputType, ubs);

    SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
    SmallVector<int64_t, 4> steps(rank, 1);
    // Copy data,
    create.affine.forIE(
        lbs, ubs, steps, [&](AffineBuilder &createAffine, ValueRange indices) {
          Value x = createAffine.load(inputMemref, indices);
          createAffine.store(x, alloc, indices);
        });

    rewriter.replaceOp(detachOp, {alloc});
    return success();
  }
};

void populateLoweringZLowAttachLayoutOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ZLowAttachLayoutLowering>(typeConverter, ctx);
}

} // namespace zlow
} // namespace onnx_mlir
