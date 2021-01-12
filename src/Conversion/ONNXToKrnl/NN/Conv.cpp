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

    // Affine maps for computing the conv window.
    std::vector<AffineMap> windowAffineMaps =
        getAffineMapsForConvWindow(rewriter, /*ceilMode=*/true, isDilated);
    AffineMap windowStartMap = windowAffineMaps[0];
    AffineMap windowDimMap = windowAffineMaps[2];
    AffineMap kernelOffsetMap = windowAffineMaps[3];

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    auto resultShape = memRefType.getShape();
    auto inputOperand = operandAdaptor.X();
    auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();
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
    //           convWindowDim1= round(float(end1 - start1) / float(d2))
    //           convWindowDim2= round(float(end2 - start2) / float(d2))
    //
    //           for c = 0 .. C/group:
    //             for cw1 in range(convWindowDim1):
    //               for cw2 in range(convWindowDim2):
    //                 # indices to access the data
    //                 h1 = cw1 * d1 + start1
    //                 h2 = cw2 * d2 + start2
    //                 # indices to access the kernel
    //                 k1 = h1 - kernelOffset1
    //                 k2 = h2 - kernelOffset2
    //                 # Update the output.
    //                 R[n][kernel][r1][r2] =
    //                   D[n][g * (C / group) + c][h1][h2] * K[kernel][c][k1][k2];
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
    auto kernelsPerGroupValue =
        rewriter.create<ConstantIndexOp>(loc, kernelsPerGroup);
    auto zero = emitConstantOp(rewriter, loc, memRefType.getElementType(), 0);
    Value subchannels;
    if (kernelShape[1] < 0) {
      subchannels = rewriter.create<DimOp>(loc, kernelOperand, 1).getResult();
    } else {
      subchannels = rewriter.create<ConstantIndexOp>(loc, kernelShape[1]);
    }

    // 1. Define outer loops and emit empty optimization block:
    int64_t nOuterLoops = (group > 1) ? 3 : 2;
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
      Value kernel = outerLoops.getInductionVar(mIndex);
      if (group > 1) {
        // Middle loop is over groups and third loop is over the
        // kernel identifiers in the current group.
        AffineMap kernelMap = AffineMap::get(2, 1,
            /*gIndex=*/rewriter.getAffineDimExpr(0) *
                    /*kernelsPerGroup=*/rewriter.getAffineSymbolExpr(0) +
                /*mIndex=*/rewriter.getAffineDimExpr(1));
        kernel = rewriter.create<AffineApplyOp>(loc, kernelMap,
            ArrayRef<Value>{/*gIndex=*/outerLoops.getInductionVar(gIndex),
                /*mIndex=*/outerLoops.getInductionVar(mIndex),
                /*kernelsPerGroupValue=*/kernelsPerGroupValue});
      }

      // 2.2 Define spatial loops
      int64_t nSpatialLoops = resultShape.size() - 2;
      BuildKrnlLoop spatialLoops(rewriter, loc, nSpatialLoops);
      spatialLoops.createDefineOp();
      for (int i = 2; i < resultShape.size(); ++i)
        spatialLoops.pushBounds(0, alloc, i);

      // 2.4 Emit loop nest over output spatial dimensions.
      //   for rX = 0 .. RX
      spatialLoops.createIterateOp();
      rewriter.setInsertionPointToStart(spatialLoops.getIterateBlock());

      {
        // 3. Emit the body of the spatial loop nest.
        // 3.1 Emit: R[n][kernel][r1][r2] = 0;
        SmallVector<Value, 4> resultIndices;
        // n
        resultIndices.emplace_back(outerLoops.getInductionVar(nIndex));
        // kernel
        resultIndices.emplace_back(kernel);
        // rX
        for (auto arg : spatialLoops.getIterateBlock()->getArguments())
          resultIndices.emplace_back(arg);
        // Store initializer value into output location.
        rewriter.create<AffineStoreOp>(loc, zero, alloc, resultIndices);

        // Prepare induction variables and constants as arguments for the affine
        // maps.
        int kernelOffset = 2;
        SmallVector<SmallVector<Value, 4>, 4> IVsAndConstants;
        { // Construct IVsAndConstants.
          for (int i = 0; i < nSpatialLoops; ++i) {
            int j = i + kernelOffset;
            SmallVector<Value, 4> ic;
            // d0, output
            ic.emplace_back(resultIndices[j]);
            // s0, input dim
            if (inputShape[j] < 0) {
              ic.emplace_back(rewriter.create<DimOp>(loc, inputOperand, j));
            } else {
              ic.emplace_back(emitConstantOp(
                  rewriter, loc, rewriter.getIndexType(), inputShape[j]));
            }
            // s1, kernel dim
            if (kernelShape[j] < 0) {
              ic.emplace_back(rewriter.create<DimOp>(loc, kernelOperand, j));
            } else {
              ic.emplace_back(emitConstantOp(
                  rewriter, loc, rewriter.getIndexType(), kernelShape[j]));
            }
            // s2, pad dim
            ic.emplace_back(emitConstantOp(
                rewriter, loc, rewriter.getIndexType(), pads[i]));
            // s3, stride dim
            ic.emplace_back(emitConstantOp(
                rewriter, loc, rewriter.getIndexType(), strides[i]));
            // s4, dilation dim
            ic.emplace_back(emitConstantOp(rewriter, loc,
                rewriter.getIndexType(), (isDilated) ? dilations[i] : 1));
            IVsAndConstants.emplace_back(ic);
          }
        }

        // Obtain values from the affine maps.
        SmallVector<Value, 4> windowStartValues, kernelOffsetValues;
        for (int i = 0; i < nSpatialLoops; ++i) {
          Value startIndex = rewriter.create<AffineMaxOp>(
              loc, windowStartMap, IVsAndConstants[i]);
          windowStartValues.emplace_back(startIndex);
          Value offsetIndex = rewriter.create<AffineMinOp>(
              loc, kernelOffsetMap, IVsAndConstants[i]);
          kernelOffsetValues.emplace_back(offsetIndex);
        }

        // 3.2 Define inner loops.
        int64_t nInnerLoops = 1 + (kernelShape.size() - 2);
        BuildKrnlLoop innerLoops(rewriter, loc, nInnerLoops);
        innerLoops.createDefineOp();
        //   for c = 0 .. C/group
        int cIndex = innerLoops.pushBounds(0, kernelOperand, 1);
        //   for Kx = 0 .. KX
        for (int i = 2; i < kernelShape.size(); ++i)
          innerLoops.pushBounds(
              0, windowDimMap, llvm::makeArrayRef(IVsAndConstants[i - 2]));

        // 3.4 Emit inner loop nest.
        innerLoops.createIterateOp();

        // Emit the bias, if needed.
        if (hasBias) {
          auto loadResult =
              rewriter.create<AffineLoadOp>(loc, alloc, resultIndices);
          SmallVector<Value, 4> biasIndices;
          biasIndices.emplace_back(kernel);
          auto loadBias =
              rewriter.create<AffineLoadOp>(loc, biasOperand, kernel);
          auto resultWithBias =
              rewriter.create<AddFOp>(loc, loadResult, loadBias);
          // Store initializer value into output location.
          rewriter.create<AffineStoreOp>(
              loc, resultWithBias, alloc, resultIndices);
        }

        //
        rewriter.setInsertionPointToStart(innerLoops.getIterateBlock());
        {
          // 4. Emit inner loop body
          // R[n][kernel][r1][r2] =
          //   D[n][g * (C / group) + c][s1 * r1 + k1][s2 * r2 + k2] *
          //   K[kernel][c][k1][k2];

          // 4.1 Prepare indices for accesing the data tensor.
          SmallVector<Value, 4> dataIndices;
          // n
          dataIndices.emplace_back(outerLoops.getInductionVar(nIndex));
          // g * (C / group) + c
          Value channelDepth = innerLoops.getInductionVar(cIndex);
          if (group > 1) {
            AffineMap indexMap = AffineMap::get(2, 1,
                /*g=*/rewriter.getAffineDimExpr(0) *
                        /*subchannel=*/rewriter.getAffineSymbolExpr(0) +
                    /*c=*/rewriter.getAffineDimExpr(1));
            channelDepth = rewriter.create<AffineApplyOp>(loc, indexMap,
                ArrayRef<Value>{/*g=*/outerLoops.getInductionVar(gIndex),
                    /*c=*/channelDepth, /*subchannel=*/subchannels});
          }
          dataIndices.emplace_back(channelDepth);

          for (int i = 0; i < nSpatialLoops; ++i) {
            if (isDilated) {
              // hi = hp * dH + startH
              Value index = rewriter.create<MulIOp>(loc,
                  innerLoops.getInductionVar(i + 1), IVsAndConstants[i][5]);
              index = rewriter.create<AddIOp>(loc, index, windowStartValues[i]);
              dataIndices.emplace_back(index);
            } else {
              // hi = hp + startH
              Value index = rewriter.create<AddIOp>(
                  loc, innerLoops.getInductionVar(i + 1), windowStartValues[i]);
              dataIndices.emplace_back(index);
            }
          }

          // 4.2 Prepare indices for accessing the kernel tensor.
          SmallVector<Value, 4> kernelIndices;
          // kernel
          kernelIndices.emplace_back(kernel);
          // c
          kernelIndices.emplace_back(innerLoops.getInductionVar(cIndex));
          // kX
          for (int i = 0; i < kernelShape.size() - 2; ++i) {
            // Since the window at borders may be smaller than the kernel, we
            // have to shift kernel indices with a suitalbe offset.
            Value kernelIndex = rewriter.create<SubIOp>(
                loc, innerLoops.getInductionVar(i + 1), kernelOffsetValues[i]);
            kernelIndices.emplace_back(kernelIndex);
          }

          // 4.3 Compute convolution.
          auto loadData =
              rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
          auto loadKernel =
              rewriter.create<LoadOp>(loc, kernelOperand, kernelIndices);
          auto loadPartialSum =
              rewriter.create<AffineLoadOp>(loc, alloc, resultIndices);
          Value result = rewriter.create<AddFOp>(loc, loadPartialSum,
              rewriter.create<MulFOp>(loc, loadData, loadKernel));
          // 4.4 Store computed value into output location.
          rewriter.create<AffineStoreOp>(loc, result, alloc, resultIndices);
        }
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
