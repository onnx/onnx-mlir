/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Conv.cpp - Lowering Convolution Op -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Convolution Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

#define DEBUG_OPTIMIZED_OFF 0

using namespace mlir;

static int64_t ceil(int64_t a, int64_t b) {
  assert(a >= 0 && b >= 0);
  return (uint64_t)ceil(((double)a) / ((double)b));
}

struct ONNXConvOpLowering : public ConversionPattern {
  ONNXConvOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConvOp::getOperationName(), 1, ctx) {}

  void convUnoptimized(ConversionPatternRewriter &rewriter,
      IndexExprScope &topScope, ONNXConvOp &convOp,
      ONNXConvOpAdaptor &operandAdaptor, ONNXConvOpShapeHelper &shapeHelper,
      MemRefType &memRefType, Value alloc, SmallVectorImpl<int64_t> &pads,
      SmallVectorImpl<int64_t> &strides,
      SmallVectorImpl<int64_t> &dilations) const {
    auto loc = convOp.getLoc();
    bool isDilated = !dilations.empty();

    KrnlBuilder createKrnl(rewriter, loc);
    MathBuilder createMath(rewriter, loc);

    // Spatial data starts from the second dimension.
    int spatialStartIndex = 2;

    auto resultShape = memRefType.getShape();
    auto inputOperand = operandAdaptor.X();
    auto kernelOperand = operandAdaptor.W();
    auto biasOperand = operandAdaptor.B();
    bool hasBias = !biasOperand.getType().isa<NoneType>();
    int64_t groupNum = convOp.group();
    IndexExpr groupNumIE = LiteralIndexExpr(groupNum);
    Value fZero = emitConstantOp(rewriter, loc, memRefType.getElementType(), 0);

    // Bounds for output sizes: [N x M x HOut x WOut]:
    // where N is Batch Size,
    // where M is Channel Out (multiple of group num)
    // and where HOut & WOut are spacial dimension of output.
    int outputRank = shapeHelper.dimsForOutput(0).size();
    IndexExpr channelOut = shapeHelper.dimsForOutput(0)[1];
    IndexExpr channelOutPerGroup = channelOut.ceilDiv(groupNumIE);
    if (channelOut.isLiteral()) {
      assert(channelOutPerGroup.isLiteral() &&
             "expected div by const to result in a literal");
      assert(groupNum * channelOutPerGroup.getLiteral() ==
                 channelOut.getLiteral() &&
             "expected channel out (M) size to be a multiple of the number of "
             "groups");
    }
    // Bounds for input image X: [N x C x Hin x Win]:
    // where N is Batch Size,
    // where C is Channel In (multiple of group num)
    // and where Hin & Win are spacial dimension of input image.
    MemRefBoundsIndexCapture inputBounds(inputOperand);
    IndexExpr batchSize = inputBounds.getSymbol(0);
    IndexExpr channelIn = inputBounds.getSymbol(1);

    // Bounds for kernel/filter W: [M x C/group x kH x kW]:
    // where M is Channel Out,
    // where C/group is number of channel in per group,
    // given C is Channel In, group is the group size,
    // and where kH x kW are the kernel / filter size (e.g. 3x3, 1x1).
    MemRefBoundsIndexCapture kernelBounds(kernelOperand);
    IndexExpr channelInPerGroup = kernelBounds.getSymbol(1);
    if (channelIn.isLiteral()) {
      assert(channelInPerGroup.isLiteral() &&
             "expected channel in per group to be "
             "literal when channel in is literal");
      assert(
          groupNum * channelInPerGroup.getLiteral() == channelIn.getLiteral() &&
          "expected Channel In (C) size to be a multiple of number of groups");
    }
    // Build outer loop, iterating over batch x groups x kernel per groups
    bool hasGroup = groupNum > 1;
    IndexExpr iZero = LiteralIndexExpr(0);
    ValueRange outerLoops = createKrnl.defineLoops(hasGroup ? 3 : 2);
    SmallVector<IndexExpr, 3> outerLbs, outerUbs;
    if (hasGroup) {
      outerLbs = {iZero, iZero, iZero};
      outerUbs = {batchSize, groupNumIE, channelOutPerGroup};
    } else {
      // TODO: just see if having a loop from 0..1 is really that bad.
      // Removing this special case would simplify the code.
      outerLbs = {iZero, iZero};
      outerUbs = {batchSize, channelOut};
    }
    // for n = 0 .. N:
    //   for g = 0 .. group:
    //     for cOPG = 0 .. channelOutPerGroup:
    //       co = g * channeloutPerGroup + cIPG;
    createKrnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs, {},
        [&](KrnlBuilder &createKrnl, ValueRange) {
      // Compute the Channel In Indices.
      ValueRange outerIndices = createKrnl.getInductionVarValue(outerLoops);
      IndexExprScope outerScope(createKrnl);
      IndexExpr channelOutIndex;         // To access the image channel.
      IndexExpr channelOutPerGroupIndex; // To access the filter channel in.
      if (hasGroup) {
        DimIndexExpr groupIndex(outerIndices[1]);
        channelOutPerGroupIndex = DimIndexExpr(outerIndices[2]);
        channelOutIndex = groupIndex * SymbolIndexExpr(channelOutPerGroup) +
                          channelOutPerGroupIndex;
      } else {
        channelOutPerGroupIndex = DimIndexExpr(outerIndices[1]);
        channelOutIndex = channelOutPerGroupIndex;
      }
      // Iterates over the output spacial dimensions
      int spacialRank = outputRank - spatialStartIndex;
      ValueRange spacialLoops = createKrnl.defineLoops(spacialRank);
      SmallVector<IndexExpr, 3> spacialLbs, spacialUbs;
      for (int s = spatialStartIndex; s < outputRank; ++s) {
        spacialLbs.emplace_back(iZero);
        spacialUbs.emplace_back(
            SymbolIndexExpr(shapeHelper.dimsForOutput(0)[s]));
      }
      // Spacial loops.
      // for h = 0 .. HOut:
      //    for w = 0 .. WOut:
      createKrnl.iterateIE(spacialLoops, spacialLoops, spacialLbs, spacialUbs,
          {}, [&](KrnlBuilder &createKrnl, ValueRange) {
            ValueRange spatialIndices =
                createKrnl.getInductionVarValue(spacialLoops);
            IndexExprScope spacialScope(createKrnl);
#if 0
                // R[n][kernel][r1][r2] = 0;
                // TODO: needed if use reduction val?
                SmallVector<IndexExpr, 4> resAccessFunc;
                resAccessFunc.emplace_back(SymbolIndexExpr(outerIndices[0]));
                resAccessFunc.emplace_back(SymbolIndexExpr(channelOutIndex));
                for (Value s : spatialIndices)
                  resAccessFunc.emplace_back(DimIndexExpr(s));
                createKrnl.storeIE(fZero, alloc, resAccessFunc);
#endif
            // Create a local reduction value and set to zero.
            MemRefType tmpType =
                MemRefType::get({}, memRefType.getElementType());
            Value reductionVal =
                createKrnl.getBuilder().create<memref::AllocaOp>(
                    createKrnl.getLoc(), tmpType);
            createKrnl.store(fZero, reductionVal);
            // Bounds for reduction loops.
            ValueRange redLoops = = createKrnl.defineLoops(spacialRank + 1);
            SmallVector<IndexExpr, 4> redLbs, redUbs;
            // First: loop over channel in per group.
            redLbs.emplace_back(iZero);
            redUbs.emplace_back(SymbolIndexExpr(channelInPerGroup));
            // For each spacial dim, do the following.
            for (int i = 0; i < spacialRank; ++i) {
              int s = i + spatialStartIndex;
              int64_t d = (isDilated) ? dilations[i] : 1;
              IndexExpr start1 =
                  DimIndexExpr(spatialIndices[i]) * strides[i] - pad[i];
              int64_t start2 = ceil(pads[i], d) * d + 1 - pads[i];
              IndexExpr start = IndexExpr::max(start1, start2);
              IndexExpr end1 = SymbolIndexExpr(shapeHelper.dimsForOutput(0)[s]);
            }
            // for c = 0 .. C/group:
            //   for cw1 in range(end1 - start1):
            //     for cw2 in range(end2 - start2):
          });
  }

  void convOriginal(ConversionPatternRewriter &rewriter,
      IndexExprScope &ieScope, ONNXConvOp &convOp,
      ONNXConvOpAdaptor &operandAdaptor, ONNXConvOpShapeHelper &shapeHelper,
      MemRefType &memRefType, Value alloc, SmallVectorImpl<int64_t> &pads,
      SmallVectorImpl<int64_t> &strides,
      SmallVectorImpl<int64_t> &dilations) const {
      auto loc = convOp.getLoc();
      bool isDilated = !dilations.empty();

      KrnlBuilder createKrnl(rewriter, loc);
      MathBuilder createMath(rewriter, loc);

      // Spatial data starts from the second dimension.
      int spatialStartIndex = 2;

      auto resultShape = memRefType.getShape();
      auto inputOperand = operandAdaptor.X();
      auto kernelOperand = operandAdaptor.W();
      auto kernelShape = kernelOperand.getType().cast<MemRefType>().getShape();
      auto biasOperand = operandAdaptor.B();
      bool hasBias = !biasOperand.getType().isa<NoneType>();

      // R = Conv(D, K)
      //
      // The input/output shapes will look like this:
      //
      // D (NxCxHxW) x K (Mx C/group x KH x KW) -> R (NxMxRHxRW)
      //
      // where C & M are also known as is Channel In (C) & Out (M)
      // also, C is a multiple of the number of groups:
      //   C = group * kernelsPerGroup
      //
      // The loop nest will look as follows:
      //
      // strides = [s1, s2]
      // dilations = [d1, d2]
      // pads = [pt1, pt2, pb1, pb2]
      //
      // kernelsPerGroup = M / group;
      // for n = 0 .. N:
      //   for g = 0 .. group:
      //     for m = 0 .. kernelsPerGroup:
      //       kernel = g * kernelsPerGroup + m; // Channel out
      //       for r1 = 0 .. RH:
      //         for r2 = 0 .. RW:
      //           R[n][kernel][r1][r2] = 0;
      //
      //           # Compute the convolution window.
      //           firstValid1 = ceil(float(pt1 / d1)) * d1 - pt1
      //           start1 = max(firstValid1, r1 * s1 - pt1)
      //           end1 = min(H, r1 * s1 + (KH -1) * d1  + 1 - pt1)
      //           kernelOffset1 = min(0, r1 * s1 - pt1)
      //
      //           firstValid2= ceil(float(p2 / d2)) * d2 - pt2
      //           start2 = max(firstValid2, r2 * s2 - pt2)
      //           end2 = min(W, r2 * s2 + (KW - 1) * d2 + 1 - pt2)
      //           kernelOffset2 = min(0, r2 * s2 - pt2)
      //
      //           for c = 0 .. C/group:
      //             for cw1 in range(end1 - start1):
      //               for cw2 in range(end2 - start2):
      //                 # indices to access the data
      //                 h1 = cw1 * d1 + start1
      //                 h2 = cw2 * d2 + start2
      //                 # indices to access the kernel
      //                 k1 = h1 - kernelOffset1
      //                 k2 = h2 - kernelOffset2
      //                 # Update the output.
      //                 R[n][kernel][r1][r2] =
      //                   D[n][g * (C / group) + c][h1][h2] *
      //                   K[kernel][c][k1][k2];
      //
      // Naming:
      //   n, g, m: outer loop nest indices
      //   r1, r2: spatial loop nest indices
      //   c, k1, k2: inner loop nest indices
      //
      //
      // In the general case:
      //
      // D (NxCxD1xD2x...xDdim) x K (MxC/groupxK1xK2x...xKdim)
      //     -> R (NxMxR1xR2x...xRdim)
      //
      // The above loop nest can be adapted by increasing the number
      // of r- and k-index loop i.e. r1 r2 and k1 k2 loops.

      // Set up outermost loops: n g m r1 r2 ... rdim
      // Skip g if group is 1.

      // Before we start the iteration we need to compute the number of
      // unsplit kernels and fetch the number of groups from the attribute
      // list. Group is always a compilation constant.
      int64_t group = convOp.group();
      // Compute the number of unsplit kernels. The number of kernels
      // must be a multiple of the number of groups.
      assert(kernelShape[0] > 0 &&
             "kernel shape is expected to be constant in code below");
      int64_t kernelsPerGroup = floor(kernelShape[0] / group);
      LiteralIndexExpr kernelsPerGroupValue(kernelsPerGroup);
      auto zero = emitConstantOp(rewriter, loc, memRefType.getElementType(), 0);
      MemRefBoundsIndexCapture kernelBounds(kernelOperand);
      DimIndexExpr subchannels(kernelBounds.getDim(1));

      // 1. Define outer loops and emit empty optimization block:
      int64_t nOuterLoops =
          (group > 1) ? (spatialStartIndex + 1) : spatialStartIndex;
      BuildKrnlLoop outerLoops(rewriter, loc, nOuterLoops);
      outerLoops.createDefineOp();
      //   for n = 0 .. N:
      int nIndex = outerLoops.pushBounds(0, inputOperand, 0);
      //   for g = 0 .. N:
      int gIndex = -1;
      if (group > 1)
        gIndex = outerLoops.pushBounds(0, group);
      //   for m = 0 .. kernelsPerGroup:
      int mIndex = outerLoops.pushBounds(0, kernelsPerGroup);
      // Outer loop iterations.
      outerLoops.createIterateOp();

      rewriter.setInsertionPointToStart(outerLoops.getIterateBlock());
      {
        // 2. Emit the body of the outer loop nest.

        // 2.1 Compute kernel order number: kernel = g * kernelsPerGroup + m;
        // If group is not set then the value of the kernel ID is
        // identical to that of the loop over kernels.
        IndexExpr kernel = DimIndexExpr(outerLoops.getInductionVar(mIndex));
        if (group > 1) {
          DimIndexExpr g(outerLoops.getInductionVar(gIndex));
          kernel = g * kernelsPerGroupValue + kernel;
        }
        // Evaluate kernel to emit its SSA value at this location.
        kernel.getValue();

        // 2.2 Define spatial loops
        int64_t nSpatialLoops = resultShape.size() - spatialStartIndex;
        BuildKrnlLoop spatialLoops(rewriter, loc, nSpatialLoops);
        spatialLoops.createDefineOp();
        for (int i = spatialStartIndex; i < (int)resultShape.size(); ++i)
          spatialLoops.pushBounds(0, alloc, i);

        // 2.4 Emit loop nest over output spatial dimensions.
        //   for rX = 0 .. RX
        spatialLoops.createIterateOp();
        rewriter.setInsertionPointToStart(spatialLoops.getIterateBlock());
        {
          // 3. Emit the body of the spatial loop nest.
          // 3.1 Emit: R[n][kernel][r1][r2] = 0;
          SmallVector<IndexExpr, 4> resultIndices;
          // n
          resultIndices.emplace_back(
              DimIndexExpr(outerLoops.getInductionVar(nIndex)));
          // kernel
          resultIndices.emplace_back(kernel);
          // rX
          for (auto arg : spatialLoops.getIterateBlock()->getArguments())
            resultIndices.emplace_back(DimIndexExpr(arg));

          // Initialize the output.
          createKrnl.storeIE(zero, alloc, resultIndices);

          // Create a local reduction value.
          Value reductionVal = rewriter.create<memref::AllocaOp>(
              loc, MemRefType::get({}, memRefType.getElementType()));
          createKrnl.store(zero, reductionVal);

          // Prepare induction variables.
          SmallVector<SmallVector<IndexExpr, 4>, 4> IVExprs;
          {
            MemRefBoundsIndexCapture inputBounds(inputOperand);
            for (int i = 0; i < nSpatialLoops; ++i) {
              int j = i + spatialStartIndex;
              SmallVector<IndexExpr, 4> ic;
              // d0, output
              ic.emplace_back(resultIndices[j]);
              // s0, input dim
              ic.emplace_back(inputBounds.getDim(j));
              // s1, kernel dim
              ic.emplace_back(kernelBounds.getDim(j));
              // s2, pad dim
              ic.emplace_back(LiteralIndexExpr(pads[i]));
              // s3, stride dim
              ic.emplace_back(LiteralIndexExpr(strides[i]));
              // s4, dilation dim
              ic.emplace_back(LiteralIndexExpr((isDilated) ? dilations[i] : 1));
              IVExprs.emplace_back(ic);
            }
          }

          // IndexExprs to compute:
          // - the start position of the conv window, and
          // - the relative offset of the kernel to the conv window's start
          // position.
          SmallVector<IndexExpr, 4> windowStartExprs, kernelOffsetExprs;
          for (int i = 0; i < nSpatialLoops; ++i) {
            std::vector<mlir::IndexExpr> exprs = getIndexExprsForConvWindow(
                IVExprs[i], /*ceilMode=*/true, isDilated);
            windowStartExprs.emplace_back(exprs[0]);
            kernelOffsetExprs.emplace_back(exprs[2]);
          }

          // 3.2 Define inner loops.
          int64_t nInnerLoops = 1 + (kernelShape.size() - spatialStartIndex);
          BuildKrnlLoop innerLoops(rewriter, loc, nInnerLoops);
          innerLoops.createDefineOp();
          //   for c = 0 .. C/group
          int cIndex = innerLoops.pushBounds(0, kernelOperand, 1);
          //   for cw1 in range(end1 - start1):
          //     for cw2 in range(end2 - start2):
          AffineMap windowSizeMap =
              getWindowAffineMap(rewriter, /*ceilMode=*/true, isDilated);
          for (int i = spatialStartIndex; i < (int)kernelShape.size(); ++i) {
            // Affine map's operands.
            SmallVector<Value, 4> operands;
            for (IndexExpr expr : IVExprs[i - spatialStartIndex])
              operands.emplace_back(expr.getValue());
            innerLoops.pushBounds(0, windowSizeMap, operands);
          }

          // 3.4 Emit inner loop nest.
          innerLoops.createIterateOp();

          //
          auto ipOuterLoopRegion = rewriter.saveInsertionPoint();
          rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());
          {
            // 4. Emit inner loop body
            //    # indices to access the data
            //    h1 = cw1 * d1 + start1
            //    h2 = cw2 * d2 + start2
            //    # indices to access the kernel
            //    k1 = h1 - kernelOffset1
            //    k2 = h2 - kernelOffset2
            //    R[n][kernel][r1][r2] =
            //      D[n][g * (C / group) + c][h1][h2] *
            //      K[kernel][c][k1][k2];

            // 4.1 Prepare indices for accesing the data tensor.
            SmallVector<IndexExpr, 4> dataIndices;
            // n
            dataIndices.emplace_back(
                DimIndexExpr(outerLoops.getInductionVar(nIndex)));
            // g * (C / group) + c
            IndexExpr channelDepth =
                DimIndexExpr(innerLoops.getInductionVar(cIndex));
            if (group > 1) {
              DimIndexExpr g(outerLoops.getInductionVar(gIndex));
              channelDepth = g * subchannels + channelDepth;
            }
            dataIndices.emplace_back(channelDepth);
            // h1 = cw1 * d1 + start1
            for (int i = 0; i < nSpatialLoops; ++i) {
              DimIndexExpr cw1(innerLoops.getInductionVar(i + 1));
              IndexExpr start1 = windowStartExprs[i];
              if (isDilated) {
                // h1 = cw1 * d1 + start1
                IndexExpr d1 = IVExprs[i][5];
                dataIndices.emplace_back(cw1 * d1 + start1);
              } else {
                // h1 = cw1 + start1
                dataIndices.emplace_back(cw1 + start1);
              }
            }

            // 4.2 Prepare indices for accessing the kernel tensor.
            // SmallVector<Value, 4> kernelIndices;
            SmallVector<IndexExpr, 4> kernelIndices;
            // kernel
            kernelIndices.emplace_back(kernel);
            // c
            kernelIndices.emplace_back(
                DimIndexExpr(innerLoops.getInductionVar(cIndex)));
            // k1 = h1 - kernelOffset1
            for (int i = 0; i < (int)kernelShape.size() - spatialStartIndex;
                 ++i) {
              // Since the window at borders may be smaller than the kernel, we
              // have to shift kernel indices with a suitable offset.
              DimIndexExpr h1(innerLoops.getInductionVar(i + 1));
              kernelIndices.emplace_back(h1 - kernelOffsetExprs[i]);
            }

            // 4.3 Compute convolution.
            auto loadData = createKrnl.loadIE(inputOperand, dataIndices);
            auto loadKernel = createKrnl.loadIE(kernelOperand, kernelIndices);
            auto loadPartialSum = createKrnl.load(reductionVal);
            Value result = createMath.add(
                loadPartialSum, createMath.mul(loadData, loadKernel));
            // 4.4 Store computed value into output location.
            createKrnl.store(result, reductionVal);
          }
          rewriter.restoreInsertionPoint(ipOuterLoopRegion);

          auto result = createKrnl.load(reductionVal);
          // Store the result. Optionally add bias.
          if (hasBias) {
            SmallVector<IndexExpr, 4> biasIndices;
            biasIndices.emplace_back(kernel);
            auto loadBias = createKrnl.loadIE(biasOperand, biasIndices);
            auto resultWithBias = createMath.add(result, loadBias);
            createKrnl.storeIE(resultWithBias, alloc, resultIndices);
          } else
            createKrnl.storeIE(result, alloc, resultIndices);
        }
      }
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
      auto loc = op->getLoc();
      ONNXConvOpAdaptor operandAdaptor(operands);
      ONNXConvOp convOp = llvm::dyn_cast<ONNXConvOp>(op);

      // Read dilations attribute if the op has.
      SmallVector<int64_t, 4> dilations;
      auto dilationsAttribute = convOp.dilationsAttr();
      bool isDefaultDilations = true;
      for (auto dilation : dilationsAttribute.getValue()) {
        int64_t dilationValue = dilation.cast<IntegerAttr>().getInt();
        if (dilationValue > 1 && isDefaultDilations)
          isDefaultDilations = false;
        dilations.emplace_back(dilationValue);
      }
      if (isDefaultDilations)
        dilations.clear();

      // Read pads attribute
      SmallVector<int64_t, 4> pads;
      auto padsAttribute = convOp.padsAttr();
      for (Attribute pad : padsAttribute.getValue())
        pads.emplace_back(pad.cast<IntegerAttr>().getInt());

      // Read strides attribute
      SmallVector<int64_t, 4> strides;
      auto stridesAttribute = convOp.stridesAttr();
      for (Attribute stride : stridesAttribute.getValue())
        strides.emplace_back(stride.cast<IntegerAttr>().getInt());

      // Get shape.
      ONNXConvOpShapeHelper shapeHelper(&convOp, rewriter,
          getDenseElementAttributeFromKrnlValue,
          loadDenseElementArrayValueAtIndex);
      auto shapecomputed =
          shapeHelper.Compute(operandAdaptor, convOp.kernel_shape(),
              padsAttribute, stridesAttribute, convOp.dilations());
      assert(succeeded(shapecomputed));

      // Scope for krnl ops
      IndexExprScope ieScope(rewriter, loc);

      // Insert an allocation and deallocation for the result of this operation.
      MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
      Value alloc = insertAllocAndDeallocSimple(
          rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

      if (DEBUG_OPTIMIZED_OFF) {
        convOriginal(rewriter, ieScope, convOp, operandAdaptor, shapeHelper,
            memRefType, alloc, pads, strides, dilations);
      } else {
        convUnoptimized(rewriter, ieScope, convOp, operandAdaptor, shapeHelper,
            memRefType, alloc, pads, strides, dilations);
      }
      rewriter.replaceOp(op, alloc);
      return success();
  }
  };

  void populateLoweringONNXConvOpPattern(
      RewritePatternSet &patterns, MLIRContext *ctx) {
    patterns.insert<ONNXConvOpLowering>(ctx);
  }
