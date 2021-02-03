//===--------------- Conv.cpp - Lowering Convolution Op
//--------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Convolution Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

std::vector<int64_t> getDilations(ONNXConvOp poolOp) {
  std::vector<int64_t> dilations;
  auto dilationsAttribute = poolOp.dilationsAttr();
  bool isDefaultDilations = true;
  for (auto dilation : dilationsAttribute.getValue()) {
    int64_t dilationValue = dilation.cast<IntegerAttr>().getInt();
    if (dilationValue > 1 && isDefaultDilations)
      isDefaultDilations = false;
    dilations.emplace_back(dilationValue);
  }
  if (isDefaultDilations)
    return {};
  else
    return dilations;
}

struct ONNXConvOpLowering : public ConversionPattern {
  ONNXConvOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConvOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ONNXConvOpAdaptor operandAdaptor(operands);
    ONNXConvOp convOp = llvm::dyn_cast<ONNXConvOp>(op);

    // Read dilations attribute if the op has.
    std::vector<int64_t> dilations = getDilations(convOp);
    bool isDilated = !dilations.empty();

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

    // Context for IndexExpr.
    IndexExprContext ieContext(&rewriter, loc);

    // Spatial data starts from the second dimension.
    int spatialStartIndex = 2;

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    auto resultShape = memRefType.getShape();
    auto inputOperand = operandAdaptor.X();
    auto kernelOperand = operandAdaptor.W();
    auto kernelShape = kernelOperand.getType().cast<MemRefType>().getShape();
    auto biasOperand = operandAdaptor.B();
    bool hasBias = !biasOperand.getType().isa<NoneType>();

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {inputOperand});

    // R = Conv(D, K)
    //
    // The input/output shapes will look like this:
    //
    // D (NxCxHxW) x K (MxC/groupxKHxKW) -> R (NxMxRHxRW)
    //
    // M is a multiple of the number of groups:
    //   M = group * kernelsPerGroup
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
    //       kernel = g * kernelsPerGroup + m;
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
    int64_t kernelsPerGroup = floor(kernelShape[0] / group);
    IndexExpr kernelsPerGroupValue =
        ieContext.createLiteralIndex(kernelsPerGroup);
    auto zero = emitConstantOp(rewriter, loc, memRefType.getElementType(), 0);
    IndexExpr subchannels =
        ieContext.createDimIndexFromShapedType(kernelOperand, 1);

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
      IndexExpr kernel = ieContext.createLoopInductionIndex(
          outerLoops.getInductionVar(mIndex));
      if (group > 1) {
        IndexExpr g = ieContext.createLoopInductionIndex(
            outerLoops.getInductionVar(gIndex));
        kernel = g * kernelsPerGroupValue + kernel;
      }

      // 2.2 Define spatial loops
      int64_t nSpatialLoops = resultShape.size() - spatialStartIndex;
      BuildKrnlLoop spatialLoops(rewriter, loc, nSpatialLoops);
      spatialLoops.createDefineOp();
      for (int i = spatialStartIndex; i < resultShape.size(); ++i)
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
        resultIndices.emplace_back(ieContext.createLoopInductionIndex(
            outerLoops.getInductionVar(nIndex)));
        // kernel
        resultIndices.emplace_back(kernel);
        // rX
        for (auto arg : spatialLoops.getIterateBlock()->getArguments())
          resultIndices.emplace_back(ieContext.createLoopInductionIndex(arg));

        // Explicitly evalutate IndexExprs. Otherwise it fails when using
        // 'resultIndices' for createKrnlStoreOp called after the reduction
        // loop.
        for (auto &ie : resultIndices)
          ie.getValue();

        // Create a local reduction value.
        Value reductionVal = rewriter.create<AllocaOp>(
            loc, MemRefType::get({}, memRefType.getElementType()));
        rewriter.create<KrnlStoreOp>(
            loc, zero, reductionVal, ArrayRef<Value>{});

        // Prepare induction variables.
        SmallVector<SmallVector<IndexExpr, 4>, 4> IVExprs;
        {
          for (int i = 0; i < nSpatialLoops; ++i) {
            int j = i + spatialStartIndex;
            SmallVector<IndexExpr, 4> ic;
            // d0, output
            ic.emplace_back(resultIndices[j]);
            // s0, input dim
            ic.emplace_back(
                ieContext.createDimIndexFromShapedType(inputOperand, j));
            // s1, kernel dim
            ic.emplace_back(
                ieContext.createDimIndexFromShapedType(kernelOperand, j));
            // s2, pad dim
            ic.emplace_back(ieContext.createLiteralIndex(pads[i]));
            // s3, stride dim
            ic.emplace_back(ieContext.createLiteralIndex(strides[i]));
            // s4, dilation dim
            ic.emplace_back(
                ieContext.createLiteralIndex((isDilated) ? dilations[i] : 1));
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
              ieContext, IVExprs[i], /*ceilMode=*/true, isDilated);
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
        for (int i = spatialStartIndex; i < kernelShape.size(); ++i) {
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
          dataIndices.emplace_back(ieContext.createLoopInductionIndex(
              outerLoops.getInductionVar(nIndex)));
          // g * (C / group) + c
          IndexExpr channelDepth = ieContext.createLoopInductionIndex(
              innerLoops.getInductionVar(cIndex));
          if (group > 1) {
            IndexExpr g = ieContext.createLoopInductionIndex(
                outerLoops.getInductionVar(gIndex));
            channelDepth = g * subchannels + channelDepth;
          }
          dataIndices.emplace_back(channelDepth);
          // h1 = cw1 * d1 + start1
          for (int i = 0; i < nSpatialLoops; ++i) {
            IndexExpr cw1 = ieContext.createLoopInductionIndex(
                innerLoops.getInductionVar(i + 1));
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
          kernelIndices.emplace_back(ieContext.createLoopInductionIndex(
              innerLoops.getInductionVar(cIndex)));
          // k1 = h1 - kernelOffset1
          for (int i = 0; i < kernelShape.size() - spatialStartIndex; ++i) {
            // Since the window at borders may be smaller than the kernel, we
            // have to shift kernel indices with a suitalbe offset.
            IndexExpr h1 = ieContext.createLoopInductionIndex(
                innerLoops.getInductionVar(i + 1));
            kernelIndices.emplace_back(h1 - kernelOffsetExprs[i]);
          }

          // 4.3 Compute convolution.
          auto loadData = ieContext.createKrnlLoadOp(inputOperand, dataIndices);
          auto loadKernel =
              ieContext.createKrnlLoadOp(kernelOperand, kernelIndices);
          auto loadPartialSum =
              rewriter.create<KrnlLoadOp>(loc, reductionVal, ArrayRef<Value>{});
          Value result = rewriter.create<AddFOp>(loc, loadPartialSum,
              rewriter.create<MulFOp>(loc, loadData, loadKernel));
          // 4.4 Store computed value into output location.
          rewriter.create<KrnlStoreOp>(
              loc, result, reductionVal, ArrayRef<Value>{});
        }
        rewriter.restoreInsertionPoint(ipOuterLoopRegion);

        auto result =
            rewriter.create<KrnlLoadOp>(loc, reductionVal, ArrayRef<Value>{});
        // Store the result. Optionally add bias.
        if (hasBias) {
          SmallVector<IndexExpr, 4> biasIndices;
          biasIndices.emplace_back(kernel);
          auto loadBias = ieContext.createKrnlLoadOp(biasOperand, biasIndices);
          auto resultWithBias = rewriter.create<AddFOp>(loc, result, loadBias);
          ieContext.createKrnlStoreOp(resultWithBias, alloc, resultIndices);
        } else
          ieContext.createKrnlStoreOp(result, alloc, resultIndices);
      }
    }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXConvOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLowering>(ctx);
}
