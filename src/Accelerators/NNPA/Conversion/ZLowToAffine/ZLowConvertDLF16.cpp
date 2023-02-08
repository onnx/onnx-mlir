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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
    Type i64Type = rewriter.getI64Type();

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

    // Compute the number of elements for conversion.
    IndexExpr numOfElements = one;
    for (IndexExpr ie : ubs)
      numOfElements = numOfElements * ie;
    SmallVector<IndexExpr, 1> ubs1D = {numOfElements};

    // Shapes for reshaping back and forth between 1-D and N-D .
    Value shape1D = create.mem.alignedAlloc(MemRefType::get({1}, indexType));
    create.affine.store(numOfElements.getValue(), shape1D, {zero.getValue()});
    Value shapeND = create.mem.alignedAlloc(MemRefType::get({rank}, indexType));
    for (int64_t i = 0; i < rank; ++i) {
      Value index = create.math.constantIndex(i);
      create.affine.store(ubs[i].getValue(), shapeND, {index});
    }

    // Reshape the input N-D MemRef into a 1-D MemRef.
    int64_t dim1DSize = ShapedType::kDynamic;
    if (numOfElements.isLiteral())
      dim1DSize = numOfElements.getLiteral();
    MemRefType input1DType = MemRefType::get({dim1DSize}, inputElementType);
    Value input1D = rewriter.create<memref::ReshapeOp>(
        loc, input1DType, inputMemref, shape1D);

    // Allocate an output 1-D MemRef.
    MemRefType output1DType =
        MemRefType::Builder(input1DType).setElementType(outputElementType);
    int64_t alignment = fromF32 ? 4096 : 64;
    Value output1D = insertAllocAndDeallocSimple(
        rewriter, nullptr, output1DType, loc, ubs1D, alignment);

    // SIMDize conversion between fp32 and dlf16.
    int64_t VL = 8;
    int64_t VLHalf = VL / 2;
    int64_t unrollFactor = 4;
    int64_t tileSize = 32;
    int64_t bufferAlignment = 64;

    int64_t cacheSize = tileSize * unrollFactor * VL;
    Value cacheSizeI64 = create.math.constant(i64Type, cacheSize);
    Value tileSizeI64 = create.math.constant(i64Type, tileSize);

    VectorType vecF32 = VectorType::get({VLHalf}, rewriter.getF32Type());
    VectorType vecF16 = VectorType::get({VL}, rewriter.getF16Type());
    MemRefType InCacheType = MemRefType::get({cacheSize}, inputElementType);
    MemRefType OutCacheType = MemRefType::get({cacheSize}, outputElementType);
    MemRefType InTileType = MemRefType::get({tileSize}, inputElementType);
    MemRefType OutTileType = MemRefType::get({tileSize}, outputElementType);

    create.affine.forIE(zero, numOfElements, cacheSize,
        [&](AffineBuilder &createAffine, Value startPos) {
          MultiDialectBuilder<AffineBuilder, KrnlBuilder, MathBuilder,
              MemRefBuilder, VectorBuilder>
              create(createAffine);
          Value zeroPos = zero.getValue();
          // Prepare a temp buffer for input.
          Value InTmp = create.mem.alignedAlloc(InCacheType, bufferAlignment);
          // Copy data to the temp input buffer.
          create.krnl.memcpy(InTmp, input1D, cacheSizeI64, zeroPos, startPos);
          // Prepare a temp buffer for output.
          Value OutTmp = create.mem.alignedAlloc(OutCacheType, bufferAlignment);

          // Computation loop.
          create.affine.forIE(zero, LiteralIndexExpr(cacheSize),
              unrollFactor * VL, [&](AffineBuilder &createAffine, Value idx) {
                MultiDialectBuilder<AffineBuilder, KrnlBuilder, MathBuilder,
                    MemRefBuilder, VectorBuilder>
                    create(createAffine);
                // Prepare a temp buffer for input.
                // Using alloca.
                Value InTileTmp =
                    create.mem.alignedAlloca(InTileType, bufferAlignment);
                // Copy data to the temp input buffer.
                create.krnl.memcpy(InTileTmp, InTmp, tileSizeI64, zeroPos, idx);
                // Prepare a temp buffer for output.
                Value OutTileTmp =
                    create.mem.alignedAlloca(OutTileType, bufferAlignment);
                // Manually unroll for simplicity.
                for (int64_t i = 0; i < unrollFactor; ++i) {
                  Value baseIdx = create.math.constantIndex(VL * i);
                  Value baseIdxNext =
                      create.math.constantIndex(VL * i + VLHalf);
                  if (fromF32) {
                    // F32 -> DLF16
                    // Load VL f32 values from the input into two vectors each
                    // with VLHalf f32 values.
                    Value vecF32H =
                        create.vec.load(vecF32, InTileTmp, {baseIdx});
                    Value vecF32L =
                        create.vec.load(vecF32, InTileTmp, {baseIdxNext});
                    Value vecDLF16 =
                        rewriter.create<ZLowConvertF32ToDLF16VectorOp>(
                            loc, vecF32H, vecF32L);
                    // Store VL f16 values back to the output.
                    create.vec.store(vecDLF16, OutTileTmp, {baseIdx});
                  } else {
                    // DLF16 -> F32
                    // Load VL f16 values from the input into a register.
                    Value vecDLF16 =
                        create.vec.load(vecF16, InTileTmp, {baseIdx});
                    auto convertOp =
                        rewriter.create<ZLowConvertDLF16ToF32VectorOp>(
                            loc, vecDLF16);
                    Value vecF32H = convertOp.getResult(0);
                    Value vecF32L = convertOp.getResult(1);
                    // Store f32 values back to the output.
                    create.vec.store(vecF32H, OutTileTmp, {baseIdx});
                    create.vec.store(vecF32L, OutTileTmp, {baseIdxNext});
                  }
                }
                // Copy data from the temp output buffer to the final output.
                create.krnl.memcpy(
                    OutTmp, OutTileTmp, tileSizeI64, idx, zeroPos);
              });

          // Copy data from the temp output buffer to the final output.
          create.krnl.memcpy(output1D, OutTmp, cacheSizeI64, startPos, zeroPos);

          // Value x = createAffine.load(input1D, {idx});
          // Value converted;
          // if (fromF32)
          //   converted = rewriter.create<ZLowConvertF32ToDLF16Op>(loc, x);
          // else
          //   converted = rewriter.create<ZLowConvertDLF16ToF32Op>(loc, x);
          // createAffine.store(converted, output1D, {idx});
        });

    // Reshape the output 1-D MemRef back into a N-D MemRef.
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
