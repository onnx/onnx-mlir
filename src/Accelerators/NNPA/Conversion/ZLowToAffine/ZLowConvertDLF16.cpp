/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- ZLowConvertDLF16Layout.cpp - Lower ZLowConvertDLF16LayoutOp ------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ZLowConvertDLF16LayoutOp operator.
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

class ZLowConvertDLF16Lowering
    : public OpConversionPattern<ZLowConvertDLF16Op> {
public:
  using OpConversionPattern<ZLowConvertDLF16Op>::OpConversionPattern;
  using OpAdaptor = typename ZLowConvertDLF16Op::Adaptor;

  LogicalResult matchAndRewrite(ZLowConvertDLF16Op convertOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = convertOp.getLoc();
    Value inputMemref = adaptor.getInput();
    StringRef direction = adaptor.getDirection();
    bool fromF32 = direction.equals_insensitive("from_f32") ? true : false;

    MemRefType outputType = convertOp.getOutput().getType().cast<MemRefType>();

    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForZLow> create(
        rewriter, loc);
    IndexExprScope indexScope(create.affine);

    SmallVector<IndexExpr, 4> ubs;
    create.zlowIE.getShapeAsDims(inputMemref, ubs);
    int rank = ubs.size();

    // Allocate the output buffer.
    int64_t alignment = fromF32 ? 4096 : -1;
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, convertOp.getOperation(), outputType, loc, ubs, alignment);

    SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
    SmallVector<int64_t, 4> steps(rank, 1);
    // Copy data,
    create.affine.forIE(
        lbs, ubs, steps, [&](AffineBuilder &createAffine, ValueRange indices) {
          Value x = createAffine.load(inputMemref, indices);
          Value converted;
          if (fromF32)
            converted = rewriter.create<ZLowConvertF32ToDLF16Op>(loc, x);
          else
            converted = rewriter.create<ZLowConvertDLF16ToF32Op>(loc, x);
          createAffine.store(converted, alloc, indices);
        });

    rewriter.replaceOp(convertOp, {alloc});
    return success();
  }
};

void populateLoweringZLowConvertDLF16OpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ZLowConvertDLF16Lowering>(typeConverter, ctx);
}

} // namespace zlow
} // namespace onnx_mlir
