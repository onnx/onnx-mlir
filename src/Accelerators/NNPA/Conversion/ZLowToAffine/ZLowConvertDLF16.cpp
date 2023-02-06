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

    MemRefType inputType = inputMemref.getType().cast<MemRefType>();
    Type inputElementType = inputType.getElementType();
    MemRefType outputType = convertOp.getOutput().getType().cast<MemRefType>();
    Type outputElementType = outputType.getElementType();
    Type indexType = rewriter.getIndexType();

    // Only lower this operation when its input has no layout map. In other
    // words, The MemRef was normalized in advance. We want to have the shape of
    // the raw continus data to do SIMD efficiently.
    assert(inputType.getLayout().isIdentity() && "MemRef is not normalized");
    assert(outputType.getLayout().isIdentity() && "MemRef is not normalized");

    MultiDialectBuilder<MathBuilder, AffineBuilder, IndexExprBuilderForZLow,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope indexScope(create.affine);
    IndexExpr zero = LiteralIndexExpr(0);
    IndexExpr one = LiteralIndexExpr(1);

    SmallVector<IndexExpr, 4> ubs;
    create.zlowIE.getShapeAsDims(inputMemref, ubs);
    int64_t rank = ubs.size();
    IndexExpr numOfElements = one;
    for (IndexExpr ie : ubs)
      numOfElements = numOfElements * ie;

    // Reshape the input N-D MemRef into a 1-D MemRef.
    Value shape1D = create.mem.alignedAlloca(MemRefType::get({1}, indexType));
    create.affine.store(numOfElements.getValue(), shape1D, {zero.getValue()});
    MemRefType input1DType;
    if (numOfElements.isLiteral())
      input1DType =
          MemRefType::get({numOfElements.getLiteral()}, inputElementType);
    else
      input1DType = MemRefType::get({ShapedType::kDynamic}, inputElementType);
    Value input1D = rewriter.create<memref::ReshapeOp>(
        loc, input1DType, inputMemref, shape1D);

    // Allocate an output 1-D MemRef.
    MemRefType output1DType =
        MemRefType::Builder(input1DType).setElementType(outputElementType);
    SmallVector<IndexExpr, 1> shape1DIE = {numOfElements};
    int64_t alignment = fromF32 ? 4096 : -1;
    Value output1D = insertAllocAndDeallocSimple(
        rewriter, nullptr, output1DType, loc, shape1DIE, alignment);

    // Allocate the output buffer.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, convertOp.getOperation(), outputType, loc, ubs, alignment);

    // Copy data using 1-D MemRefs.
    IndexExpr lbs = zero;
    int64_t step = 1;
    // Copy data,
    create.affine.forIE(lbs, numOfElements, step,
        [&](AffineBuilder &createAffine, Value index) {
          Value x = createAffine.load(input1D, {index});
          Value converted;
          if (fromF32)
            converted = rewriter.create<ZLowConvertF32ToDLF16Op>(loc, x);
          else
            converted = rewriter.create<ZLowConvertDLF16ToF32Op>(loc, x);
          createAffine.store(converted, output1D, {index});
        });

    // int64_t tileSize = 8;
    // SmallVector<AffineForOp, 2> tiledLoops;
    // SmallVector<AffineForOp, 1> loopsToTile{};
    // if (failed(tilePerfectlyNested(loopsToTile, tileSize, &tiledLoops))) {
    //   return failure();
    // }

    // Reshape the output 1-D MemRef back into a N-D MemRef.
    Value shapeND =
        create.mem.alignedAlloca(MemRefType::get({rank}, indexType));
    for (int64_t i = 0; i < rank; ++i) {
      Value index = create.math.constantIndex(i);
      create.affine.store(ubs[i].getValue(), shapeND, {index});
    }
    Value outputND =
        rewriter.create<memref::ReshapeOp>(loc, outputType, output1D, shapeND);

    rewriter.replaceOp(convertOp, {outputND});
    return success();
  }
};

void populateLoweringZLowConvertDLF16OpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ZLowConvertDLF16Lowering>(typeConverter, ctx);
}

} // namespace zlow
} // namespace onnx_mlir
