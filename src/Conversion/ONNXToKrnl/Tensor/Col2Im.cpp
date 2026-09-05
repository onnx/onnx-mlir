/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Col2Im.cpp - Lowering Col2Im Op ------------------===//
//
// This file lowers the ONNX Col2Im Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// Read a spatial-rank sized int64 array attribute, filling in `defaultVal`
// when the attribute is absent.
static SmallVector<int64_t, 4> getSpatialAttrOrDefault(
    std::optional<ArrayAttr> attr, int64_t spatialRank, int64_t defaultVal) {
  SmallVector<int64_t, 4> vals(spatialRank, defaultVal);
  if (attr.has_value())
    for (int64_t i = 0; i < spatialRank; ++i)
      vals[i] = mlir::cast<IntegerAttr>((*attr)[i]).getInt();
  return vals;
}

struct ONNXCol2ImOpLowering : public OpConversionPattern<ONNXCol2ImOp> {
  ONNXCol2ImOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXCol2ImOp::getOperationName());
  }

  bool enableParallel = false;

  LogicalResult matchAndRewrite(ONNXCol2ImOp col2ImOp,
      ONNXCol2ImOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = col2ImOp.getOperation();
    Location loc = ONNXLoc<ONNXCol2ImOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value input = adaptor.getInput();
    Value blockShape = adaptor.getBlockShape();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder, SCFBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXCol2ImOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr outputDims = shapeHelper.getOutputDims();
    int64_t outRank = outputDims.size();
    int64_t spatialRank = outRank - 2;

    // Attributes (all static, given default values per the ONNX spec).
    SmallVector<int64_t, 4> dilations =
        getSpatialAttrOrDefault(col2ImOp.getDilations(), spatialRank, 1);
    SmallVector<int64_t, 4> strides =
        getSpatialAttrOrDefault(col2ImOp.getStrides(), spatialRank, 1);
    SmallVector<int64_t, 4> padsBegin(spatialRank, 0), padsEnd(spatialRank, 0);
    if (std::optional<ArrayAttr> pads = col2ImOp.getPads()) {
      for (int64_t i = 0; i < spatialRank; ++i) {
        padsBegin[i] = mlir::cast<IntegerAttr>((*pads)[i]).getInt();
        padsEnd[i] = mlir::cast<IntegerAttr>((*pads)[spatialRank + i]).getInt();
      }
    }
    for (int64_t i = 0; i < spatialRank; ++i)
      if (strides[i] <= 0)
        return op->emitOpError("strides must be strictly positive");

    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    Type elementType = mlir::cast<MemRefType>(convertedType).getElementType();
    if (!elementType.isIntOrIndexOrFloat())
      return op->emitOpError(
          "Col2Im lowering only supports integer and floating point element "
          "types");

    // Block shape values (spatial), as (possibly dynamic) index expressions.
    SmallVector<IndexExpr, 4> blockDims;
    for (int64_t i = 0; i < spatialRank; ++i)
      blockDims.emplace_back(
          create.krnlIE.getIntFromArrayAsSymbol(blockShape, i));

    // Suffix products of the block dims, used to flatten the (k_1..k_n)
    // block-local index into the flat "column" index (row-major order).
    assert(spatialRank >= 1 && "Col2Im requires at least one spatial dim");
    SmallVector<IndexExpr, 4> blockSuffixProd(spatialRank, LiteralIndexExpr(1));
    for (int64_t i = spatialRank - 2; i >= 0; --i)
      blockSuffixProd[i] = blockSuffixProd[i + 1] * blockDims[i + 1];
    IndexExpr blockProd = blockSuffixProd[0] * blockDims[0];

    // Grid dims: number of blocks placed along each spatial axis, following
    // the same formula as a convolution's output spatial size.
    SmallVector<IndexExpr, 4> gridDims;
    for (int64_t i = 0; i < spatialRank; ++i) {
      IndexExpr d = outputDims[2 + i];
      IndexExpr numerator = d + LiteralIndexExpr(padsBegin[i] + padsEnd[i]) -
                            LiteralIndexExpr(dilations[i]) *
                                (blockDims[i] - LiteralIndexExpr(1)) -
                            LiteralIndexExpr(1);
      gridDims.emplace_back(
          numerator.floorDiv(strides[i]) + LiteralIndexExpr(1));
    }
    // Suffix products of grid dims, used to flatten the (l_1..l_n) block
    // position index into the flat "L" index (row-major order).
    SmallVector<IndexExpr, 4> gridSuffixProd(spatialRank, LiteralIndexExpr(1));
    for (int64_t i = spatialRank - 2; i >= 0; --i)
      gridSuffixProd[i] = gridSuffixProd[i + 1] * gridDims[i + 1];

    // Allocate the output.
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Value alloc = create.mem.alignedAlloc(outputMemRefType, outputDims);

    // Outer loop: one iteration per output element (n, c, x_1, .., x_n).
    // Each iteration writes exactly one output element and owns its own
    // scratch accumulator, so the outer loop is safe to parallelize.
    ValueRange outerLoopDef = create.krnl.defineLoops(outRank);
    SmallVector<IndexExpr, 4> outerLbs(outRank, LiteralIndexExpr(0));
    if (enableParallel)
      tryCreateKrnlParallel(create.krnl, op, "col2im outer loop", outerLoopDef,
          outerLbs, outputDims, 0, 2, {}, /*min iter for going parallel*/ 16);
    create.krnl.iterateIE(outerLoopDef, outerLoopDef, outerLbs, outputDims,
        [&](const KrnlBuilder &createKrnl, ValueRange outerLoopInd) {
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
              MemRefBuilder, MathBuilder, SCFBuilder>
              create(createKrnl);

          SmallVector<IndexExpr, 4> outputIndices;
          for (int64_t i = 0; i < outRank; ++i)
            outputIndices.emplace_back(DimIndexExpr(outerLoopInd[i]));

          // Local accumulator for output[n, c, x_1, .., x_n].
          Value reductionVal =
              create.mem.alloca(MemRefType::get({}, elementType));
          create.krnl.store(create.math.constant(elementType, 0), reductionVal);

          // c * blockProd is invariant across the whole (k_1..k_n) sweep
          // below; hoist it out so it is computed once per output element
          // instead of once per valid (k_1..k_n) hit.
          IndexExpr cTimesBlockProd = outputIndices[1] * blockProd;

          // Inner loop: for every position (k_1, .., k_n) within a block,
          // find whether it maps back to this output element, and if so
          // accumulate the corresponding column entry.
          ValueRange innerLoopDef = create.krnl.defineLoops(spatialRank);
          SmallVector<IndexExpr, 4> innerLbs(spatialRank, LiteralIndexExpr(0));
          create.krnl.iterateIE(innerLoopDef, innerLoopDef, innerLbs, blockDims,
              [&](const KrnlBuilder &createKrnl2, ValueRange innerLoopInd) {
                MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                    MemRefBuilder, MathBuilder, SCFBuilder>
                    create(createKrnl2);

                SmallVector<IndexExpr, 4> kIdx;
                for (int64_t i = 0; i < spatialRank; ++i)
                  kIdx.emplace_back(DimIndexExpr(innerLoopInd[i]));

                // For each spatial axis: x_i = l_i * stride_i + k_i *
                // dilation_i - padBegin_i.  Solve for l_i and validate.
                SmallVector<IndexExpr, 4> lIdx(spatialRank);
                Value validCond = nullptr;
                for (int64_t i = 0; i < spatialRank; ++i) {
                  IndexExpr numerator =
                      outputIndices[2 + i] + LiteralIndexExpr(padsBegin[i]) -
                      kIdx[i] * LiteralIndexExpr(dilations[i]);
                  Value numV = numerator.getValue();
                  Value zeroV = create.math.constantIndex(0);
                  Value strideV = create.math.constantIndex(strides[i]);
                  Value geZero = create.math.sge(numV, zeroV);
                  Value remV = create.math.rem(numV, strideV);
                  Value divisible = create.math.eq(remV, zeroV);
                  Value lV = create.math.floorDiv(numV, strideV);
                  lIdx[i] = DimIndexExpr(lV);
                  Value ltGrid = create.math.slt(lV, gridDims[i].getValue());
                  Value dimValid = create.math.andi(geZero, divisible);
                  dimValid = create.math.andi(dimValid, ltGrid);
                  validCond = validCond ? create.math.andi(validCond, dimValid)
                                        : dimValid;
                }
                if (!validCond)
                  validCond = create.math.constant(rewriter.getI1Type(), 1);

                create.scf.ifThenElse(validCond, [&](const SCFBuilder &b) {
                  MultiDialectBuilder<KrnlBuilder, MathBuilder> create(b);

                  // Flatten (k_1..k_n) and (l_1..l_n) into the input's
                  // "C*prod(block_shape)" and "L" dimensions respectively.
                  IndexExpr flatK = LiteralIndexExpr(0);
                  IndexExpr flatL = LiteralIndexExpr(0);
                  for (int64_t i = 0; i < spatialRank; ++i) {
                    flatK = flatK + kIdx[i] * blockSuffixProd[i];
                    flatL = flatL + lIdx[i] * gridSuffixProd[i];
                  }
                  IndexExpr dim1 = cTimesBlockProd + flatK;

                  SmallVector<IndexExpr, 3> inputIndices{
                      outputIndices[0], dim1, flatL};
                  Value inputVal = create.krnl.loadIE(input, inputIndices);
                  Value cur = create.krnl.load(reductionVal);
                  create.krnl.store(
                      create.math.add(cur, inputVal), reductionVal);
                });
              });

          Value finalVal = create.krnl.load(reductionVal);
          create.krnl.storeIE(finalVal, alloc, outputIndices);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXCol2ImOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXCol2ImOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
