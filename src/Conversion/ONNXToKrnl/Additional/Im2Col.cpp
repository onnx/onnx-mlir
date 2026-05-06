/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- Im2Col.cpp - Lowering ONNXIm2ColOp ---------------------------===//
//
// Copyright 2026 The IBM Research Authors.
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
    IndexExpr N = create.krnlIE.getShapeAsDim(input, 0);
    IndexExpr CI = create.krnlIE.getShapeAsDim(input, 1);

    // Get spatial input dimensions.
    SmallVector<IndexExpr, 4> inputSpatialDims;
    for (int64_t i = 0; i < spatialRank; ++i) {
      inputSpatialDims.push_back(create.krnlIE.getShapeAsDim(input, 2 + i));
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
          // General: q = oh * (OW * OD * ...) + ow * (OD * ...) + od * ... +
          // ...
          IndexExpr q = LitIE(0);
          IndexExpr stride = LitIE(1);
          for (int64_t i = spatialRank - 1; i >= 0; --i) {
            q = q + outputIndices[i] * stride;
            if (i > 0)
              stride = stride * DimIE(outputSpatialDims[i]);
          }

          // Compute row index p = ci * KH * KW + kh * KW + kw (for 2D case).
          // General: p = ci * (KH * KW * ...) + kh * (KW * ...) + kw * ... +
          // ...
          IndexExpr p = ci;
          for (int64_t i = 0; i < spatialRank; ++i) {
            p = p * LitIE(kernelShape[i]);
          }
          for (int64_t i = 0; i < spatialRank; ++i) {
            int64_t kernelStride = 1;
            for (int64_t j = i + 1; j < spatialRank; ++j) {
              kernelStride *= kernelShape[j];
            }
            p = p + kernelIndices[i] * LitIE(kernelStride);
          }

          // Compute input spatial indices with stride/dilation/padding.
          SmallVector<IndexExpr, 4> inputSpatialIndices;
          for (int64_t i = 0; i < spatialRank; ++i) {
            IndexExpr oi = outputIndices[i];
            IndexExpr ki = kernelIndices[i];
            IndexExpr si = LitIE(strides[i]);
            IndexExpr di = LitIE(dilations[i]);
            IndexExpr padBefore = DimIE(pads[i]);
            IndexExpr inputIdx = oi * si + ki * di - padBefore;
            inputSpatialIndices.push_back(inputIdx);
          }

          // Check bounds for all spatial dimensions.
          Value inbounds = create.math.constant(rewriter.getI1Type(), true);
          for (int64_t i = 0; i < spatialRank; ++i) {
            IndexExpr inputIdx = inputSpatialIndices[i];
            IndexExpr inputDim = DimIE(inputSpatialDims[i]);

            Value geZero = create.math.sge(
                inputIdx.getValue(), create.math.constantIndex(0));
            Value ltDim =
                create.math.slt(inputIdx.getValue(), inputDim.getValue());
            Value dimInbounds = create.math.andi(geZero, ltDim);
            inbounds = create.math.andi(inbounds, dimInbounds);
          }

          // Load value if in bounds, otherwise use zero.
          Value zero = create.math.constant(inputType.getElementType(), 0.0);
          Value zeroIndex = LitIE(0).getValue();

          SmallVector<Value, 5> inputIndicesVals;
          inputIndicesVals.push_back(n.getValue());
          inputIndicesVals.push_back(ci.getValue());
          for (int64_t i = 0; i < spatialRank; ++i) {
            // Access could be out of bound; in which case indices can be set to
            // zero, don't care as result will not be used.
            Value index = inputSpatialIndices[i].getValue();
            Value inboundIndex = create.math.select(inbounds, index, zeroIndex);
            inputIndicesVals.push_back(inboundIndex);
          }

          Value actualValue = create.krnl.load(input, inputIndicesVals);
          Value loadedValue = create.math.select(inbounds, actualValue, zero);

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
  // 2D optimized path:
  // - One large outer loop over linearized columns n * (OH * OW).
  // - SCF if/else to split interior and border cases.
  // - SCF loops inside each branch.
  // - Same output layout [N, CI*KH*KW, OH*OW] as the simple version.
  //===----------------------------------------------------------------------===//

  void optimizedCodeGen(Operation *op, ConversionPatternRewriter &rewriter,
      Value input, Value alloc, ONNXIm2ColOpShapeHelper &shapeHelper) const {
    Location loc = ONNXLoc<ONNXIm2ColOp>(op);
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder, SCFBuilder>
        create(rewriter, loc);

    // Get input shape.
    MemRefType inputType = mlir::cast<MemRefType>(input.getType());
    int64_t inputRank = inputType.getRank();
    int64_t spatialRank = inputRank - 2;

    // Optimized implementation is currently specialized for 2D.
    if (spatialRank != 2) {
      simpleCodeGen(op, rewriter, input, alloc, shapeHelper);
      return;
    }

    // Get attributes from shape helper.
    const auto &kernelShape = shapeHelper.kernelShape;
    const auto &strides = shapeHelper.strides;
    const auto &dilations = shapeHelper.dilations;
    const auto &pads = shapeHelper.pads;
    const auto &outputSpatialDims = shapeHelper.outputSpatialDims;

    // Get dimensions. Any use in nested scopes must go through DimIE(value).
    IndexExpr N = create.krnlIE.getShapeAsDim(input, 0);
    IndexExpr CI = create.krnlIE.getShapeAsDim(input, 1);
    IndexExpr HIn = create.krnlIE.getShapeAsDim(input, 2);
    IndexExpr WIn = create.krnlIE.getShapeAsDim(input, 3);
    IndexExpr OH = outputSpatialDims[0];
    IndexExpr OW = outputSpatialDims[1];

    int64_t KH = kernelShape[0];
    int64_t KW = kernelShape[1];
    int64_t strideH = strides[0];
    int64_t strideW = strides[1];
    int64_t dilationH = dilations[0];
    int64_t dilationW = dilations[1];

    IndexExpr padHBegin = pads[0];
    IndexExpr padWBegin = pads[1];

    IndexExpr outputImageSize = OH * OW;
    int64_t kernelSize = KH * KW;
    IndexExpr totalColumns = N * outputImageSize;

    Value zeroIndex = create.math.constantIndex(0);
    Value zeroValue = create.math.constant(inputType.getElementType(), 0.0);

    ValueRange outerLoopDef = create.krnl.defineLoops(1);
    DimsExpr lbs(1, LitIE(0));
    DimsExpr ubs(1, totalColumns);
    if (enableParallel) {
      tryCreateKrnlParallel(create.krnl, op, "im2col outer loop parallelized",
          outerLoopDef, lbs, ubs, 0, 1, {},
          /*min iter for going parallel*/ 8,
          /*createKrnlParallel=*/true);
    }
    create.krnl.iterateIE(outerLoopDef, outerLoopDef, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange outerInd) {
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
              SCFBuilder>
              create(createKrnl);
          IndexExprScope outerScope(create.krnlIE);

          IndexExpr linearColIE = DimIE(outerInd[0]);

          IndexExpr n = linearColIE.floorDiv(DimIE(outputImageSize));
          IndexExpr q = linearColIE - n * DimIE(outputImageSize);
          IndexExpr oh = q.floorDiv(DimIE(OW));
          IndexExpr ow = q - oh * DimIE(OW);

          // Input origin shared by the full kernel sweep for this output
          // column.
          IndexExpr ihBase = oh * strideH - DimIE(padHBegin);
          IndexExpr iwBase = ow * strideW - DimIE(padWBegin);

          IndexExpr lastIH = ihBase + ((KH - 1) * dilationH);
          IndexExpr lastIW = iwBase + ((KW - 1) * dilationW);
          Value isInterior = create.math.andi(
              create.math.andi(create.math.sge(ihBase.getValue(), zeroIndex),
                  create.math.sge(iwBase.getValue(), zeroIndex)),
              create.math.andi(
                  create.math.slt(lastIH.getValue(), DimIE(HIn).getValue()),
                  create.math.slt(lastIW.getValue(), DimIE(WIn).getValue())));

          create.scf.ifThenElse(
              isInterior,
              [&](const SCFBuilder &scfThen) {
                MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                    MathBuilder>
                    create(scfThen);
                IndexExprScope thenScope(create.krnlIE);

                scfThen.forLoopIE(LitIE(0), DimIE(CI), 1,
                    /*useParallel=*/false,
                    [&](const SCFBuilder &scfCi, ValueRange ciInd) {
                      MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                          MathBuilder>
                          create(scfCi);
                      IndexExprScope ciScope(create.krnlIE);
                      IndexExpr ci = DimIE(ciInd[0]);
                      IndexExpr ciBase = ci * kernelSize;

                      scfCi.forLoopIE(LitIE(0), LitIE(KH), 1,
                          /*useParallel=*/false,
                          [&](const SCFBuilder &scfKh, ValueRange khInd) {
                            MultiDialectBuilder<KrnlBuilder,
                                IndexExprBuilderForKrnl, MathBuilder>
                                create(scfKh);
                            IndexExprScope khScope(create.krnlIE);
                            IndexExpr kh = DimIE(khInd[0]);
                            IndexExpr ih = DimIE(ihBase) + kh * dilationH;
                            IndexExpr rowBase = DimIE(ciBase) + kh * KW;

                            scfKh.forLoopIE(LitIE(0), LitIE(KW), 1,
                                /*useParallel=*/false,
                                [&](const SCFBuilder &scfKw, ValueRange kwInd) {
                                  MultiDialectBuilder<KrnlBuilder,
                                      IndexExprBuilderForKrnl, MathBuilder>
                                      create(scfKw);
                                  IndexExprScope kwScope(create.krnlIE);
                                  IndexExpr kw = DimIE(kwInd[0]);
                                  IndexExpr p = DimIE(rowBase) + kw;
                                  IndexExpr iw = DimIE(iwBase) + kw * dilationW;
                                  Value val = create.krnl.loadIE(input,
                                      {DimIE(n), DimIE(ci), DimIE(ih), iw});
                                  create.krnl.storeIE(
                                      val, alloc, {DimIE(n), p, DimIE(q)});
                                });
                          });
                    });
              },
              [&](const SCFBuilder &scfElse) {
                MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                    MathBuilder>
                    create(scfElse);
                IndexExprScope elseScope(create.krnlIE);

                IndexExpr hInInnerIE = DimIE(HIn);
                IndexExpr wInInnerIE = DimIE(WIn);

                scfElse.forLoopIE(LitIE(0), DimIE(CI), 1,
                    /*useParallel=*/false,
                    [&](const SCFBuilder &scfCi, ValueRange ciInd) {
                      MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                          MathBuilder>
                          create(scfCi);
                      IndexExprScope ciScope(create.krnlIE);
                      IndexExpr ci = DimIE(ciInd[0]);
                      IndexExpr ciBase = ci * kernelSize;

                      scfCi.forLoopIE(LitIE(0), LitIE(KH), 1,
                          /*useParallel=*/false,
                          [&](const SCFBuilder &scfKh, ValueRange khInd) {
                            MultiDialectBuilder<KrnlBuilder,
                                IndexExprBuilderForKrnl, MathBuilder>
                                create(scfKh);
                            IndexExprScope khScope(create.krnlIE);
                            IndexExpr kh = DimIE(khInd[0]);
                            IndexExpr ih = DimIE(ihBase) + kh * dilationH;
                            IndexExpr rowBase = DimIE(ciBase) + kh * KW;
                            // Compute KH-dependent bounds checks once.
                            Value ihInbounds = create.math.andi(
                                create.math.sge(
                                    DimIE(ih).getValue(), zeroIndex),
                                create.math.slt(DimIE(ih).getValue(),
                                    hInInnerIE.getValue()));
                            Value inboundIH = create.math.select(
                                ihInbounds, DimIE(ih).getValue(), zeroIndex);

                            scfKh.forLoopIE(LitIE(0), LitIE(KW), 1,
                                /*useParallel=*/false,
                                [&](const SCFBuilder &scfKw, ValueRange kwInd) {
                                  MultiDialectBuilder<KrnlBuilder,
                                      IndexExprBuilderForKrnl, MathBuilder>
                                      create(scfKw);
                                  IndexExprScope kwScope(create.krnlIE);
                                  IndexExpr kw = DimIE(kwInd[0]);
                                  IndexExpr p = DimIE(rowBase) + kw;
                                  IndexExpr iw = DimIE(iwBase) + kw * dilationW;
                                  // Compute KW-dependent bounds checks.
                                  Value iwInbounds = create.math.andi(
                                      create.math.sge(
                                          DimIE(iw).getValue(), zeroIndex),
                                      create.math.slt(DimIE(iw).getValue(),
                                          wInInnerIE.getValue()));
                                  Value inbounds =
                                      create.math.andi(ihInbounds, iwInbounds);
                                  // Now create iw that is guaranteed to be
                                  // inbounds.
                                  Value inboundIW = create.math.select(inbounds,
                                      DimIE(iw).getValue(), zeroIndex);
                                  // Load the actual value when inbound, inbound
                                  // junk otherwise.
                                  Value actualValue = create.krnl.load(
                                      input, {DimIE(n).getValue(),
                                                 DimIE(ci).getValue(),
                                                 inboundIH, inboundIW});
                                  // Select actual value or zero (padding).
                                  Value val = create.math.select(
                                      inbounds, actualValue, zeroValue);
                                  // Store result.
                                  create.krnl.storeIE(
                                      val, alloc, {DimIE(n), p, DimIE(q)});
                                });
                          });
                    });
              });
        });
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
  patterns.insert<ONNXIm2ColOpLowering>(
      typeConverter, ctx, enableParallel, useOptimizedAlgo);
}

} // namespace onnx_mlir
