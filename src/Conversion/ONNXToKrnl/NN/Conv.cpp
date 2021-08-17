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

#define DEBUG_ORIGINAL 0
#define DEBUG_TRACE 0

using namespace mlir;

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

    // Spatial data starts from the second dimension.
    int spatialStartIndex = 2;

    auto inputOperand = operandAdaptor.X();
    auto filterOperand = operandAdaptor.W();
    auto biasOperand = operandAdaptor.B();
    bool hasBias = !biasOperand.getType().isa<NoneType>();
    int64_t groupNum = convOp.group();
    IndexExpr G = LiteralIndexExpr(groupNum);
    Value fZero = emitConstantOp(rewriter, loc, memRefType.getElementType(), 0);

    // Bounds for output sizes: [N x CO x HO x WO]:
    // where N is Batch Size,
    // where CO (or M) is Channel Out (multiple of group num)
    // and where HO & WO are spacial dimensions of the output.
    int outputRank = shapeHelper.dimsForOutput().size();
    IndexExpr N = shapeHelper.dimsForOutput()[0];
    IndexExpr CO = shapeHelper.dimsForOutput()[1];
    IndexExpr COPerGroup = CO.ceilDiv(G);
    // Note: Pytorch requires both channel in (CI) and channel out (CO) to be
    // multiple of group number (G).
    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    // ONNX clearly states that C (channel in or CI here) is a multiple of group
    // number (G).
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
    // Quote: X.shape[1] == (W.shape[1] * group) == C
    // Keras also specifies it: Input channels and filters must both be
    // divisible by groups.
    // https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    if (CO.isLiteral()) {
      assert(COPerGroup.isLiteral() &&
             "expected div by const to result in a literal");
      assert(groupNum * COPerGroup.getLiteral() == CO.getLiteral() &&
             "expected Channel Out (M) size to be a multiple of group num");
    }

    // Bounds for input image X: [N x CI x HI x WI]:
    // where N is Batch Size,
    // where CI (or C) is Channel In (multiple of group num),
    // and where HI & WI are spacial dimensions of the input image.
    MemRefBoundsIndexCapture inputBounds(inputOperand);
    IndexExpr CI = inputBounds.getSymbol(1);

    // Bounds for kernel/filter W: [CO x CIPerGroup x KH x KW]:
    // where CO (or M) is Channel Out,
    // where CIPerGroup (or C/G) is number of channel in per group,
    // and where KH x KW are the kernel / filter size (e.g. 3x3, 1x1).
    MemRefBoundsIndexCapture filterBounds(filterOperand);
    IndexExpr CIPerGroup = filterBounds.getSymbol(1);
    if (CI.isLiteral()) {
      assert(CIPerGroup.isLiteral() &&
             "expected channel in per group to be literal when CI is literal");
      assert(groupNum * CIPerGroup.getLiteral() == CI.getLiteral() &&
             "expected Channel In (C) size to be a multiple of group num");
    }

    // Determine the bounds for the loops over batch & channel out.
    IndexExpr iZero = LiteralIndexExpr(0);
    ValueRange outerLoops = createKrnl.defineLoops(3);
    SmallVector<IndexExpr, 3> outerLbs = {iZero, iZero, iZero};
    SmallVector<IndexExpr, 3> outerUbs = {N, G, COPerGroup};
    // Iterate over the outer loops
    // for n = 0 .. N:
    //   for g = 0 .. G:
    //     for coPerGroup = 0 .. COPerGroup:
    //       co = g * COPerGroup + coPerGroup;

    createKrnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs, {},
        [&](KrnlBuilder &createKrnl, ValueRange) {
          // Compute the Channel In Indices.
          ValueRange outerIndices = createKrnl.getInductionVarValue(outerLoops);
          IndexExprScope outerScope(createKrnl);
          // Compute the channel out index "co".
          DimIndexExpr g(outerIndices[1]);
          DimIndexExpr coPerGroup(outerIndices[2]);
          IndexExpr co = g * SymbolIndexExpr(COPerGroup) + coPerGroup;
          // Compute g * CIPerGroup for later use.
          IndexExpr gTimesCIPerGroup = g * SymbolIndexExpr(CIPerGroup);
          // Determine the bounds for the output spacial dimensions.
          int spacialRank = outputRank - spatialStartIndex;
          ValueRange outputSpacialLoops = createKrnl.defineLoops(spacialRank);
          SmallVector<IndexExpr, 3> outputSpacialLbs, outputSpacialUbs;
          for (int i = spatialStartIndex; i < outputRank; ++i) {
            outputSpacialLbs.emplace_back(iZero);
            outputSpacialUbs.emplace_back(
                SymbolIndexExpr(shapeHelper.dimsForOutput()[i]));
          }
          // Spacial loops.
          // for ho = 0 .. HO:
          //    for wo = 0 .. WO:
          createKrnl.iterateIE(outputSpacialLoops, outputSpacialLoops,
              outputSpacialLbs, outputSpacialUbs, {},
              [&](KrnlBuilder &createKrnl, ValueRange) {
                ValueRange outputSpatialIndices =
                    createKrnl.getInductionVarValue(outputSpacialLoops);
                IndexExprScope outputSpacialScope(createKrnl);

                // Create a local reduction value and set to zero.
                MemRefType tmpType =
                    MemRefType::get({}, memRefType.getElementType());
                Value reductionVal =
                    createKrnl.getBuilder().create<memref::AllocaOp>(
                        createKrnl.getLoc(), tmpType);
                createKrnl.store(fZero, reductionVal);

                // Bounds for reduction loops.
                ValueRange redLoops = createKrnl.defineLoops(spacialRank + 1);
                SmallVector<IndexExpr, 4> redLbs, redUbs, pMinOS;
                // First: loop over channel in per group.
                redLbs.emplace_back(iZero);
                redUbs.emplace_back(SymbolIndexExpr(CIPerGroup));
                // For each spacial dim, do the following.
                for (int i = 0; i < spacialRank; ++i) {
                  // Get data for dis spacial dimension.
                  DimIndexExpr o(outputSpatialIndices[i]);
                  SymbolIndexExpr I(
                      inputBounds.getSymbol(spatialStartIndex + i));
                  SymbolIndexExpr K(
                      filterBounds.getSymbol(spatialStartIndex + i));
                  LiteralIndexExpr p(pads[i]); // Begining/left/top pad.
                  LiteralIndexExpr s(strides[i]);
                  LiteralIndexExpr d(isDilated ? dilations[i] : 1);
                  // lb = ceil((p - o * s) / d)
                  IndexExpr pos = p - (o * s);
                  IndexExpr lb = pos.ceilDiv(d);
                  lb = IndexExpr::max(lb, 0);
                  redLbs.emplace_back(lb);
                  // ub = ceil((I + p - o * s) / d)
                  IndexExpr ipos = I + pos;
                  IndexExpr ub = ipos.ceilDiv(d);
                  ub = IndexExpr::min(ub, K);
                  redUbs.emplace_back(ub);
                  // Save p - o * s for later use.
                  pMinOS.emplace_back(pos);
                }
                // for ciPerGroup = 0 .. CIPerGroup:
                //   for kh in lb .. ub:
                //     for kw in lb .. ub:
                createKrnl.iterateIE(redLoops, redLoops, redLbs, redUbs, {},
                    [&](KrnlBuilder &createKrnl, ValueRange) {
                      ValueRange redIndices =
                          createKrnl.getInductionVarValue(redLoops);
                      IndexExprScope redScope(createKrnl);
                      MathBuilder createMath(createKrnl);
                      // Create access function for input image:
                      // [n, ci, ho * sh + kh * dh - ph, wo * sw + kw * dw -
                      // pw].
                      SmallVector<IndexExpr, 4> inputAccessFct;
                      DimIndexExpr n(outerIndices[0]);
                      inputAccessFct.emplace_back(n);
                      // ci = g * CIPerG + ciPerG
                      DimIndexExpr ciPerG(redIndices[0]);
                      IndexExpr ci = SymbolIndexExpr(gTimesCIPerGroup) + ciPerG;
                      inputAccessFct.emplace_back(ci);
                      for (int i = 0; i < spacialRank; ++i) {
                        // for each spacial dims: access is o * s + k * d - p.
                        DimIndexExpr k(redIndices[1 + i]);
                        SymbolIndexExpr pos(pMinOS[i]);
                        LiteralIndexExpr d(isDilated ? dilations[i] : 1);
                        // k*d - (p - o*s) = k*d + o*s - p
                        IndexExpr t = (k * d) - pos;
                        inputAccessFct.emplace_back(t);
                      }
                      Value image =
                          createKrnl.loadIE(inputOperand, inputAccessFct);
                      // Create access fct for filter: [co, ciPerG, kh, kw].
                      SmallVector<IndexExpr, 4> filterAccessFct;
                      filterAccessFct.emplace_back(DimIndexExpr(co));
                      filterAccessFct.emplace_back(DimIndexExpr(ciPerG));

                      for (int i = 0; i < spacialRank; ++i) {
                        DimIndexExpr k(redIndices[1 + i]);
                        filterAccessFct.emplace_back(k);
                      }
                      Value filter =
                          createKrnl.loadIE(filterOperand, filterAccessFct);
                      Value oldRed = createKrnl.load(reductionVal);
                      Value mul = createMath.mul(image, filter);
                      Value newRed = createMath.add(oldRed, mul);
                      createKrnl.store(newRed, reductionVal);
                    }); // Reduction loops.
                        // Finish the reduction and store in result array.
                Value result = createKrnl.load(reductionVal);
                // Store the result. Optionally add bias.
                SymbolIndexExpr coInOutputSpacial(co);
                if (hasBias) {
                  MathBuilder createMath(createKrnl);
                  Value bias =
                      createKrnl.loadIE(biasOperand, {coInOutputSpacial});
                  result = createMath.add(result, bias);
                }
                SmallVector<IndexExpr, 4> resAccessFunc;
                resAccessFunc.emplace_back(SymbolIndexExpr(outerIndices[0]));
                resAccessFunc.emplace_back(coInOutputSpacial);
                for (Value o : outputSpatialIndices)
                  resAccessFunc.emplace_back(DimIndexExpr(o));
                createKrnl.storeIE(result, alloc, resAccessFunc);
              }); // Output spacial loops.
        });       // Outer loops;
  }

  // Not needed anymore, kept for reference for the near future.
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

    // Insert an allocation and deallocation for the result of this operation.
    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

    if (DEBUG_ORIGINAL) {
      convOriginal(rewriter, shapeHelper.scope, convOp, operandAdaptor,
          shapeHelper, memRefType, alloc, pads, strides, dilations);
    } else {
      convUnoptimized(rewriter, shapeHelper.scope, convOp, operandAdaptor,
          shapeHelper, memRefType, alloc, pads, strides, dilations);
    }
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXConvOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLowering>(ctx);
}
