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

    // Get output spatial dimensions.
    const auto &outputSpatialDims = shapeHelper.outputSpatialDims;

    // Compute strides for row and column indexing.
    SmallVector<IndexExpr, 4> outputStrides;
    IndexExpr outputStride = LitIE(1);
    for (int64_t i = spatialRank - 1; i >= 0; --i) {
      outputStrides.insert(outputStrides.begin(), outputStride);
      outputStride = outputStride * outputSpatialDims[i];
    }

    SmallVector<IndexExpr, 4> kernelStrides;
    IndexExpr kernelStride = LitIE(1);
    for (int64_t i = spatialRank - 1; i >= 0; --i) {
      kernelStrides.insert(kernelStrides.begin(), kernelStride);
      kernelStride = kernelStride * LitIE(kernelShape[i]);
    }

    // Create loops: n, o1, o2, ..., oN, ci, k1, k2, ..., kN.
    int64_t numLoops = 1 + spatialRank + 1 + spatialRank;
    ValueRange loopDef = create.krnl.defineLoops(numLoops);

    // Set up loop bounds.
    SmallVector<IndexExpr, 8> lbs(numLoops, LitIE(0));
    SmallVector<IndexExpr, 8> ubs;
    ubs.push_back(N); // n
    for (int64_t i = 0; i < spatialRank; ++i) {
      ubs.push_back(outputSpatialDims[i]); // o1, o2, ..., oN
    }
    ubs.push_back(CI); // ci
    for (int64_t i = 0; i < spatialRank; ++i) {
      ubs.push_back(LitIE(kernelShape[i])); // k1, k2, ..., kN
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

          // Compute output row index.
          IndexExpr row = n * SymIE(outputStride);
          for (int64_t i = 0; i < spatialRank; ++i) {
            row = row + outputIndices[i] * SymIE(outputStrides[i]);
          }

          // Compute output column index.
          IndexExpr col = ci * SymIE(kernelStride);
          for (int64_t i = 0; i < spatialRank; ++i) {
            col = col + kernelIndices[i] * SymIE(kernelStrides[i]);
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

          // Store to output[row, col].
          SmallVector<Value, 2> outputIndicesVals;
          outputIndicesVals.push_back(row.getValue());
          outputIndicesVals.push_back(col.getValue());
          create.krnl.store(loadedValue, alloc, outputIndicesVals);
        });
  }

  //===----------------------------------------------------------------------===//
  // Optimized code generation for Im2Col operation.
  //===----------------------------------------------------------------------===//

  void optimizedCodeGen(Operation *op, ConversionPatternRewriter &rewriter,
      Value input, Value alloc, ONNXIm2ColOpShapeHelper &shapeHelper) const {
    Location loc = ONNXLoc<ONNXIm2ColOp>(op);
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

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
      reshapedDims.push_back(LitIE(k));
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
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
              SCFBuilder>
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

          // Check if all elements in the receptive field are in bounds.
          // For each spatial dimension i, check:
          //   minIdx[i] = oi * stride[i] - pad[i] >= 0
          //   maxIdx[i] = oi * stride[i] + (kernelShape[i]-1) * dilation[i] -
          //   pad[i] < inputDim[i]
          Value allInBounds = create.math.constant(rewriter.getI1Type(), true);
          for (int64_t i = 0; i < spatialRank; ++i) {
            IndexExpr oi = SymIE(outputIndices[1 + i]);
            IndexExpr si = LitIE(strides[i]);
            IndexExpr di = LitIE(dilations[i]);
            IndexExpr padBefore = SymIE(pads[i]);
            IndexExpr ki = LitIE(kernelShape[i] - 1);
            IndexExpr inputDim = SymIE(inputSpatialDims[i]);

            // Compute min and max input indices for this dimension.
            IndexExpr minIdx = oi * si - padBefore;
            IndexExpr maxIdx = oi * si + ki * di - padBefore;

            // Check bounds: minIdx >= 0 && maxIdx < inputDim.
            Value minGeZero = create.math.sge(
                minIdx.getValue(), create.math.constantIndex(0));
            Value maxLtDim =
                create.math.slt(maxIdx.getValue(), inputDim.getValue());
            Value dimInBounds = create.math.andi(minGeZero, maxLtDim);
            allInBounds = create.math.andi(allInBounds, dimInBounds);
          }

          // Use if-then-else to create two versions of the inner loops.
          int64_t numFieldLoops = 1 + spatialRank;
          SmallVector<IndexExpr, 5> fieldLbs(numFieldLoops, LitIE(0));
          SmallVector<IndexExpr, 5> fieldUbs;
          fieldUbs.push_back(SymIE(CI));
          for (int64_t i = 0; i < spatialRank; ++i) {
            fieldUbs.push_back(LitIE(kernelShape[i]));
          }

          create.scf.ifThenElse(
              allInBounds,
              [&](const SCFBuilder &createSCF) {
                // Fast path: All elements in bounds, no checks needed.
                MultiDialectBuilder<SCFBuilder, IndexExprBuilderForKrnl,
                    MathBuilder, MemRefBuilder>
                    create(createSCF);

                // Use SCF loops for the inner loops.
                SmallVector<int64_t> steps(numFieldLoops, 1);
                SmallVector<bool> useParallel(numFieldLoops, false);
                create.scf.forLoopsIE(fieldLbs, fieldUbs, steps, useParallel,
                    [&](const SCFBuilder &createSCF2, ValueRange fieldLoopInd) {
                      MultiDialectBuilder<SCFBuilder, IndexExprBuilderForKrnl,
                          MathBuilder, MemRefBuilder>
                          create(createSCF2);
                      IndexExprScope innerScope(createSCF2);

                      // Extract ci and kernel indices.
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
                        IndexExpr padBefore = SymIE(pads[i]);
                        IndexExpr inputIdx = oi * si + ki * di - padBefore;
                        inputSpatialIndices.push_back(inputIdx);
                      }

                      // Load value directly (no bounds check).
                      SmallVector<Value, 5> inputIndicesVals;
                      inputIndicesVals.push_back(n.getValue());
                      inputIndicesVals.push_back(ci.getValue());
                      for (int64_t i = 0; i < spatialRank; ++i) {
                        inputIndicesVals.push_back(
                            inputSpatialIndices[i].getValue());
                      }

                      Value loadedValue =
                          create.mem.load(input, inputIndicesVals);

                      // Store to reshaped output.
                      SmallVector<Value, 6> outputIndicesVals;
                      outputIndicesVals.push_back(rowIdx.getValue());
                      outputIndicesVals.push_back(ci.getValue());
                      for (int64_t i = 0; i < spatialRank; ++i) {
                        outputIndicesVals.push_back(
                            kernelIndices[i].getValue());
                      }
                      create.mem.store(
                          loadedValue, reshapedAlloc, outputIndicesVals);
                    });
              },
              [&](const SCFBuilder &createSCF) {
                // Slow path: Need bounds checks for each element.
                MultiDialectBuilder<SCFBuilder, IndexExprBuilderForKrnl,
                    MathBuilder, MemRefBuilder>
                    create(createSCF);

                // Use SCF loops for the inner loops.
                SmallVector<int64_t> steps(numFieldLoops, 1);
                SmallVector<bool> useParallel(numFieldLoops, false);
                create.scf.forLoopsIE(fieldLbs, fieldUbs, steps, useParallel,
                    [&](const SCFBuilder &createSCF2, ValueRange fieldLoopInd) {
                      MultiDialectBuilder<SCFBuilder, IndexExprBuilderForKrnl,
                          MathBuilder, MemRefBuilder>
                          create(createSCF2);
                      IndexExprScope innerScope(createSCF2);

                      // Extract ci and kernel indices.
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
                        IndexExpr padBefore = SymIE(pads[i]);
                        IndexExpr inputIdx = oi * si + ki * di - padBefore;
                        inputSpatialIndices.push_back(inputIdx);
                      }

                      // Check bounds and load value.
                      Value zero =
                          create.math.constant(inputType.getElementType(), 0.0);

                      // Build condition for all spatial dimensions being in
                      // bounds.
                      Value inBounds =
                          create.math.constant(rewriter.getI1Type(), true);
                      for (int64_t i = 0; i < spatialRank; ++i) {
                        IndexExpr inputIdx = inputSpatialIndices[i];
                        IndexExpr inputDim = SymIE(inputSpatialDims[i]);

                        Value geZero = create.math.sge(
                            inputIdx.getValue(), create.math.constantIndex(0));
                        Value ltDim = create.math.slt(
                            inputIdx.getValue(), inputDim.getValue());
                        Value dimInBounds = create.math.andi(geZero, ltDim);
                        inBounds = create.math.andi(inBounds, dimInBounds);
                      }

                      // Load value if in bounds.
                      SmallVector<Value, 5> inputIndicesVals;
                      inputIndicesVals.push_back(n.getValue());
                      inputIndicesVals.push_back(ci.getValue());
                      for (int64_t i = 0; i < spatialRank; ++i) {
                        inputIndicesVals.push_back(
                            inputSpatialIndices[i].getValue());
                      }

                      Value actualValue =
                          create.mem.load(input, inputIndicesVals);
                      Value loadedValue =
                          create.math.select(inBounds, actualValue, zero);

                      // Store to reshaped output.
                      SmallVector<Value, 6> outputIndicesVals;
                      outputIndicesVals.push_back(rowIdx.getValue());
                      outputIndicesVals.push_back(ci.getValue());
                      for (int64_t i = 0; i < spatialRank; ++i) {
                        outputIndicesVals.push_back(
                            kernelIndices[i].getValue());
                      }
                      create.mem.store(
                          loadedValue, reshapedAlloc, outputIndicesVals);
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
    assert(outputRank == 2 && "Im2Col output must be 2D");

    // Insert an allocation for the output.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    // Compute output dimensions using shape helper.
    ONNXIm2ColOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Allocate with 2D shape [numRows, numCols].
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
      useOptimizedAlgo && true /* hi alex, switch to force naive algo */);
}

} // namespace onnx_mlir

// Made with Bob
