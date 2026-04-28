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

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXIm2ColOpLowering : public OpConversionPattern<ONNXIm2ColOp> {
  ONNXIm2ColOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      bool enableParallel, bool optimizedAlgo)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXIm2ColOp::getOperationName());
    this->optimizedAlgo = optimizedAlgo;
  }

  bool enableParallel = false;
  bool optimizedAlgo = false;

  //===----------------------------------------------------------------------===//
  // Simple straightforward code generation for Im2Col operation.
  // Based on the paper: "A Simple and Efficient Implementation of im2col in
  // Convolution Neural Networks" by Hao Zhang.
  // Output shape: [N, CI*KH*KW, OH*OW]
  // Loop order: N, OH, OW, ..., CI, KH, KW, ...
  // Row index p encodes (ci, kh, kw): p = ci * KH * KW + kh * KW + kw
  // Column index q encodes (oh, ow): q = oh * OW + ow
  //===----------------------------------------------------------------------===//

  void simpleCodeGen(Operation *op, ConversionPatternRewriter &rewriter,
      Value input, Value alloc, ONNXIm2ColOpShapeHelper &shapeHelper) const {
    Location loc = ONNXLoc<ONNXIm2ColOp>(op);
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    // Get input shape.
    MemRefType inputType = mlir::cast<MemRefType>(input.getType());
    int64_t inputRank = inputType.getRank();
    int64_t spatialRank = inputRank - 2;

    // Get attributes from shape helper.
    const auto &kernelShape = shapeHelper.kernelShape;
    const auto &strides = shapeHelper.strides;
    const auto &dilations = shapeHelper.dilations;
    const auto &pads = shapeHelper.pads;

    // Get dimensions.
    IndexExpr N = create.krnlIE.getShapeAsSymbol(input, 0);
    IndexExpr CI = create.krnlIE.getShapeAsSymbol(input, 1);

    // Get spatial input dimensions.
    SmallVector<IndexExpr, 4> inputSpatialDims;
    for (int64_t i = 0; i < spatialRank; ++i) {
      inputSpatialDims.push_back(create.krnlIE.getShapeAsSymbol(input, 2 + i));
    }

    // Get output spatial dimensions from shape helper.
    const auto &outputSpatialDims = shapeHelper.outputSpatialDims;

    // Create nested loops: n, oh, ow, ..., ci, kh, kw, ...
    // Loop order matches the paper: N, output_spatial_dims, CI, kernel_shape.
    int64_t numLoops = 1 + spatialRank + 1 + spatialRank;
    ValueRange loopDef = create.krnl.defineLoops(numLoops);

    // Set up loop bounds.
    SmallVector<IndexExpr, 8> lbs(numLoops, LitIE(0));
    SmallVector<IndexExpr, 8> ubs;
    ubs.push_back(N); // n
    for (int64_t i = 0; i < spatialRank; ++i) {
      ubs.push_back(outputSpatialDims[i]); // oh, ow, ...
    }
    ubs.push_back(CI); // ci
    for (int64_t i = 0; i < spatialRank; ++i) {
      ubs.push_back(LitIE(kernelShape[i])); // kh, kw, ...
    }

    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder>
              create(createKrnl);
          IndexExprScope scope(createKrnl);

          // Extract loop indices.
          IndexExpr n = DimIE(loopInd[0]);
          SmallVector<IndexExpr, 4> outputIndices;
          for (int64_t i = 0; i < spatialRank; ++i) {
            outputIndices.push_back(DimIE(loopInd[1 + i]));
          }
          IndexExpr ci = DimIE(loopInd[1 + spatialRank]);
          SmallVector<IndexExpr, 4> kernelIndices;
          for (int64_t i = 0; i < spatialRank; ++i) {
            kernelIndices.push_back(DimIE(loopInd[2 + spatialRank + i]));
          }

          // Compute column index q = oh * OW + ow (for 2D case).
          // General: q = oh * (OW * OD * ...) + ow * (OD * ...) + od * ... + ...
          IndexExpr q = LitIE(0);
          IndexExpr stride = LitIE(1);
          for (int64_t i = spatialRank - 1; i >= 0; --i) {
            q = q + outputIndices[i] * stride;
            if (i > 0)
              stride = stride * DimIE(outputSpatialDims[i]);
          }

          // Compute row index p = ci * KH * KW + kh * KW + kw (for 2D case).
          // General: p = ci * (KH * KW * ...) + kh * (KW * ...) + kw * ... + ...
          IndexExpr p = ci;
          for (int64_t i = 0; i < spatialRank; ++i) {
            p = p * LitIE(kernelShape[i]);
          }
          for (int64_t i = 0; i < spatialRank; ++i) {
            IndexExpr kernelStride = LitIE(1);
            for (int64_t j = i + 1; j < spatialRank; ++j) {
              kernelStride = kernelStride * LitIE(kernelShape[j]);
            }
            p = p + kernelIndices[i] * kernelStride;
          }

          // Compute input spatial indices with stride/dilation/padding.
          SmallVector<IndexExpr, 4> inputSpatialIndices;
          for (int64_t i = 0; i < spatialRank; ++i) {
            IndexExpr oi = outputIndices[i];
            IndexExpr ki = kernelIndices[i];
            IndexExpr si = LitIE(strides[i]);
            IndexExpr di = LitIE(dilations[i]);
            IndexExpr padBefore = SymIE(pads[i]);
            IndexExpr inputIdx = oi * si + ki * di - padBefore;
            inputSpatialIndices.push_back(inputIdx);
          }

          // Check bounds for all spatial dimensions.
          Value inBounds = create.math.constant(rewriter.getI1Type(), true);
          for (int64_t i = 0; i < spatialRank; ++i) {
            IndexExpr inputIdx = inputSpatialIndices[i];
            IndexExpr inputDim = SymIE(inputSpatialDims[i]);

            Value geZero = create.math.sge(
                inputIdx.getValue(), create.math.constantIndex(0));
            Value ltDim =
                create.math.slt(inputIdx.getValue(), inputDim.getValue());
            Value dimInBounds = create.math.andi(geZero, ltDim);
            inBounds = create.math.andi(inBounds, dimInBounds);
          }

          // Load value if in bounds, otherwise use zero.
          Value zero = create.math.constant(inputType.getElementType(), 0.0);

          SmallVector<Value, 5> inputIndicesVals;
          inputIndicesVals.push_back(n.getValue());
          inputIndicesVals.push_back(ci.getValue());
          for (int64_t i = 0; i < spatialRank; ++i) {
            inputIndicesVals.push_back(inputSpatialIndices[i].getValue());
          }

          Value actualValue = create.krnl.load(input, inputIndicesVals);
          Value loadedValue = create.math.select(inBounds, actualValue, zero);

          // Store to output[n, p, q].
          SmallVector<Value, 3> outputIndicesVals;
          outputIndicesVals.push_back(n.getValue());
          outputIndicesVals.push_back(p.getValue());
          outputIndicesVals.push_back(q.getValue());
          create.krnl.store(loadedValue, alloc, outputIndicesVals);
        });
  }

  //===----------------------------------------------------------------------===//
  // Optimized code generation for Im2Col operation.
  // TODO: Re-implement optimized version based on the correct algorithm above.
  // For now, just call the simple version.
  //===----------------------------------------------------------------------===//

  void optimizedCodeGen(Operation *op, ConversionPatternRewriter &rewriter,
      Value input, Value alloc, ONNXIm2ColOpShapeHelper &shapeHelper) const {
    // Use simple code generation for now.
    // Future optimization: loop tiling, vectorization, etc.
    simpleCodeGen(op, rewriter, input, alloc, shapeHelper);
  }

  LogicalResult matchAndRewrite(ONNXIm2ColOp im2colOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = im2colOp.getOperation();
    Location loc = ONNXLoc<ONNXIm2ColOp>(op);
    ValueRange operands = adaptor.getOperands();

    // Get input.
    Value input = adaptor.getX();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getRank();
    assert(outputRank == 3 && "Im2Col output must be 3D");

    // Insert an allocation for the output.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    // Compute output dimensions using shape helper.
    ONNXIm2ColOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate with 3D shape [N, CI*KH*KW, OH*OW].
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Generate code using selected algorithm.
    if (optimizedAlgo) {
      optimizedCodeGen(op, rewriter, input, alloc, shapeHelper);
    } else {
      simpleCodeGen(op, rewriter, input, alloc, shapeHelper);
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXIm2ColOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  bool useOptimizedAlgo = OptimizationLevel > OptLevel::O0;
  patterns.insert<ONNXIm2ColOpLowering>(typeConverter, ctx, enableParallel,
      useOptimizedAlgo && false /* hi alex, switch to force naive algo */);
}

} // namespace onnx_mlir

// Made with Bob
