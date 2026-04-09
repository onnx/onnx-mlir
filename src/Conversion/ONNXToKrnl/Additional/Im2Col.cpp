/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- Im2Col.cpp - Lowering ONNXIm2ColOp ---------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Im2Col Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXIm2ColOpLowering : public OpConversionPattern<ONNXIm2ColOp> {
  ONNXIm2ColOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXIm2ColOp::getOperationName());
  }

  bool enableParallel = false;

  LogicalResult matchAndRewrite(ONNXIm2ColOp im2colOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = im2colOp.getOperation();
    Location loc = ONNXLoc<ONNXIm2ColOp>(op);

    // Get input.
    Value input = adaptor.getX();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getRank();
    assert(outputRank == 2 && "Im2Col output must be 2D");

    // Insert an allocation for the output.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    // Compute output dimensions using shape helper.
    ONNXIm2ColOpShapeHelper shapeHelper(op, {});
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate with 2D shape [numRows, numCols].
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Get input shape.
    MemRefType inputType = mlir::cast<MemRefType>(input.getType());
    int64_t inputRank = inputType.getRank();
    int64_t spatialRank = inputRank - 2;

    // Get attributes from shape helper (already computed).
    const auto &kernelShape = shapeHelper.kernelShape;
    const auto &strides = shapeHelper.strides;
    const auto &dilations = shapeHelper.dilations;
    const auto &pads =
        shapeHelper.pads; // IndexExpr, all pads [begin_0..N, end_0..N].

    // Create index expressions for dimensions.
    // hi alex , don't think we need that IndexExprScope scope(create.krnl);

    // Get N and CI dimensions.
    IndexExpr N = create.krnlIE.getShapeAsSymbol(input, 0);
    IndexExpr CI = create.krnlIE.getShapeAsSymbol(input, 1);

    // Get spatial input dimensions.
    SmallVector<IndexExpr, 4> inputSpatialDims;
    for (int64_t i = 0; i < spatialRank; ++i) {
      inputSpatialDims.push_back(create.krnlIE.getShapeAsSymbol(input, 2 + i));
    }

    // Get output spatial dimensions from shape helper (already computed).
    const auto &outputSpatialDims = shapeHelper.outputSpatialDims;

    // Get numRows from shape helper output dimensions.
    IndexExpr numRows = shapeHelper.getOutputDims()[0];

    // Reshape alloc to [numRows, CI, K1, K2, ..., KN] for easier indexing.
    DimsExpr reshapedDims;
    reshapedDims.push_back(numRows);
    reshapedDims.push_back(CI);
    for (int64_t k : kernelShape) {
      reshapedDims.push_back(LiteralIndexExpr(k));
    }

    Value reshapedAlloc = create.mem.reinterpretCast(alloc, reshapedDims);

    // Compute strides for output positions (loop invariant).
    SmallVector<IndexExpr, 4> outputStrides;
    IndexExpr stride = LitIE(1);
    for (int64_t i = spatialRank - 1; i >= 0; --i) {
      outputStrides.insert(outputStrides.begin(), stride);
      stride = stride * outputSpatialDims[i];
    }

    // Create single loop for output positions (collapsed).
    ValueRange outerLoopDef = create.krnl.defineLoops(1);

    // Enable parallelism if required.
    SmallVector<IndexExpr, 1> lbs = {LitIE(0)};
    SmallVector<IndexExpr, 1> ubs = {numRows};
    if (enableParallel)
      tryCreateKrnlParallel(
          create.krnl, op, "im2col", outerLoopDef, lbs, ubs, 16);

    create.krnl.iterateIE(outerLoopDef, outerLoopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange outerLoopInd) {
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder>
              create(createKrnl);
          IndexExprScope outerScope(createKrnl);

          // Get output row index.
          IndexExpr rowIdx = DimIE(outerLoopInd[0]);

          // Decompose rowIdx into [n, o1, o2, ..., on].
          SmallVector<IndexExpr, 5> outputIndices;
          IndexExpr remaining = rowIdx;

          // Extract n.
          IndexExpr s = SymIE(stride);
          IndexExpr n = remaining.floorDiv(s);
          outputIndices.push_back(n);
          remaining = remaining % s;

          // Extract spatial output indices.
          for (int64_t i = 0; i < spatialRank; ++i) {
            IndexExpr o = SymIE(outputStrides[i]);
            IndexExpr oi = remaining.floorDiv(o);
            outputIndices.push_back(oi);
            remaining = remaining % o;
          }

          // Create nested loops for receptive field: ci, k1, k2, ..., kN.
          int64_t numFieldLoops = 1 + spatialRank;
          ValueRange fieldLoopDef = create.krnl.defineLoops(numFieldLoops);

          SmallVector<IndexExpr, 5> fieldLbs(numFieldLoops, LitIE(0));
          SmallVector<IndexExpr, 5> fieldUbs;
          fieldUbs.push_back(SymIE(CI));
          for (int64_t i = 0; i < spatialRank; ++i) {
            fieldUbs.push_back(LitIE(kernelShape[i]));
          }

          create.krnl.iterateIE(fieldLoopDef, fieldLoopDef, fieldLbs, fieldUbs,
              [&](const KrnlBuilder &createKrnl2, ValueRange fieldLoopInd) {
                MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                    MathBuilder>
                    create(createKrnl2);
                IndexExprScope innerScope(createKrnl2);

                // Extract ci and kernel indices directly from loop variables.
                IndexExpr ci = DimIE(fieldLoopInd[0]);
                SmallVector<IndexExpr, 4> kernelIndices;
                for (int64_t i = 0; i < spatialRank; ++i) {
                  kernelIndices.push_back(DimIE(fieldLoopInd[1 + i]));
                }

                // Compute input spatial indices.
                SmallVector<IndexExpr, 4> inputSpatialIndices;
                for (int64_t i = 0; i < spatialRank; ++i) {
                  IndexExpr oi = SymIE(outputIndices[1 + i]);
                  IndexExpr ki = kernelIndices[i];
                  IndexExpr si = LitIE(strides[i]);
                  IndexExpr di = LitIE(dilations[i]);
                  IndexExpr padBefore =
                      SymIE(pads[i]); // Begin pad from outer scope.

                  IndexExpr inputIdx = oi * si + ki * di - padBefore;
                  inputSpatialIndices.push_back(inputIdx);
                }

                // Check bounds and load value.
                Value zero =
                    create.math.constant(inputType.getElementType(), 0.0);

                // Build condition for all spatial dimensions being in bounds.
                Value inBounds =
                    create.math.constant(rewriter.getI1Type(), true);
                for (int64_t i = 0; i < spatialRank; ++i) {
                  IndexExpr inputIdx = inputSpatialIndices[i];
                  IndexExpr inputDim =
                      SymIE(inputSpatialDims[i]); // From outer scope.

                  Value geZero = create.math.sge(
                      inputIdx.getValue(), create.math.constantIndex(0));
                  Value ltDim =
                      create.math.slt(inputIdx.getValue(), inputDim.getValue());
                  Value dimInBounds = create.math.andi(geZero, ltDim);
                  inBounds = create.math.andi(inBounds, dimInBounds);
                }

                // Load value if in bounds.
                SmallVector<Value, 5> inputIndicesVals;
                inputIndicesVals.push_back(n.getValue());
                inputIndicesVals.push_back(ci.getValue());
                for (int64_t i = 0; i < spatialRank; ++i) {
                  inputIndicesVals.push_back(inputSpatialIndices[i].getValue());
                }

                Value actualValue = create.krnl.load(input, inputIndicesVals);
                Value loadedValue =
                    create.math.select(inBounds, actualValue, zero);

                // Store to reshaped output[rowIdx, ci, k1, k2, ..., kN].
                SmallVector<Value, 6> outputIndicesVals;
                outputIndicesVals.push_back(rowIdx.getValue());
                outputIndicesVals.push_back(ci.getValue());
                for (int64_t i = 0; i < spatialRank; ++i) {
                  outputIndicesVals.push_back(kernelIndices[i].getValue());
                }
                create.krnl.store(
                    loadedValue, reshapedAlloc, outputIndicesVals);
              });
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXIm2ColOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXIm2ColOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir

// Made with Bob
