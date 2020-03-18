//===----- conv.cpp - Lowering Convolution Op -----------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Convolution Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/conversion/onnx_to_krnl/onnx_to_krnl_common.hpp"

using namespace mlir;

struct ONNXConvNoBiasOpLowering : public ConversionPattern {
  ONNXConvNoBiasOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConvNoBiasOp::getOperationName(), 1, ctx) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    ONNXConvNoBiasOp convOp = llvm::dyn_cast<ONNXConvNoBiasOp>(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {operands[0]});

    auto resultShape = memRefType.getShape();
    auto &inputOperand = operands[0];
    auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();
    auto &kernelOperand = operands[1];
    auto kernelShape = kernelOperand.getType().cast<MemRefType>().getShape();

    // R = ConvNoBias(D, K)
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
    //
    // kernelsPerGroup = M / group;
    // for n = 0 .. N:
    //   for g = 0 .. group:
    //     for m = 0 .. kernelsPerGroup:
    //       kernel = g * kernelsPerGroup + m;
    //       for r1 = 0 .. RH:
    //         for r2 = 0 .. RW:
    //           R[n][kernel][r1][r2] = 0;
    //           for c = 0 .. C/group:
    //             for k1 = 0 .. KH:
    //               for k2 = 0 .. KW:
    //                 R[n][kernel][r1][r2] =
    //                   D[n][g * (C / group) + c][s1 * r1 + k1][s2 * r2 + k2] *
    //                   K[kernel][c][k1][k2];
    //
    // Naming:
    //   n, g, m: outer loop nest indices
    //   r1, r2: spatial loop nest indices
    //   c, k1, k2: inner loop nest indices
    //
    // TODO: handle padding.
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
    int64_t group = convOp.group().getSExtValue();
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
    outerLoops.createDefineAndOptimizeOp();
    //   for n = 0 .. N:
    int nIndex = outerLoops.pushBounds(0, inputOperand, 0);
    //   for g = 0 .. N:
    int gIndex = -1;
    if (group > 1)
      gIndex = outerLoops.pushBounds(0, group);
    //   for m = 0 .. kernelsPerGroup:
    int mIndex = outerLoops.pushBounds(0, kernelsPerGroup);
    // Outer loop iteration
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
        auto kernelsOffset = rewriter.create<MulIOp>(
            loc, outerLoops.getInductionVar(gIndex), kernelsPerGroupValue);
        kernel = rewriter.create<AddIOp>(
            loc, kernelsOffset, outerLoops.getInductionVar(mIndex));
      }

      // 2.2 Define spatial loops
      int64_t nSpatialLoops = resultShape.size() - 2;
      BuildKrnlLoop spatialLoops(rewriter, loc, nSpatialLoops);
      spatialLoops.createDefineAndOptimizeOp();
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
        rewriter.create<StoreOp>(loc, zero, alloc, resultIndices);

        // 3.2 Define inner loops.
        int64_t nInnerLoops = 1 + (kernelShape.size() - 2);
        BuildKrnlLoop innerLoops(rewriter, loc, nInnerLoops);
        innerLoops.createDefineAndOptimizeOp();
        //   for c = 0 .. C/group
        int cIndex = innerLoops.pushBounds(0, kernelShape[1]);
        //   for Kx = 0 .. KX
        for (int i = 2; i < kernelShape.size(); ++i)
          innerLoops.pushBounds(0, kernelOperand, i);

        // 3.4 Emit inner loop nest.
        innerLoops.createIterateOp();
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
          if (group > 1)
            channelDepth = rewriter.create<AddIOp>(loc, channelDepth,
                rewriter.create<MulIOp>(
                    loc, subchannels, outerLoops.getInductionVar(gIndex)));
          dataIndices.emplace_back(channelDepth);
          // sX * rX + kX
          auto stridesAttribute = convOp.stridesAttr();
          // Read strides attribute
          SmallVector<int, 4> strides;
          if (stridesAttribute)
            for (auto stride : stridesAttribute.getValue())
              strides.emplace_back(stride.cast<IntegerAttr>().getInt());
          for (int i = 0; i < kernelShape.size() - 2; ++i) {
            Value spatialIndex = spatialLoops.getInductionVar(i);
            // If strides are present then emit the correct access index.
            if (stridesAttribute && strides[i] > 1)
              spatialIndex = rewriter.create<MulIOp>(loc,
                  rewriter.create<ConstantIndexOp>(loc, strides[i]),
                  spatialLoops.getInductionVar(i));
            dataIndices.emplace_back(rewriter.create<AddIOp>(
                loc, spatialIndex, innerLoops.getInductionVar(i + 1)));
          }

          // 4.2 Prepare indices for accessing the kernel tensor.
          SmallVector<Value, 4> kernelIndices;
          // kernel
          kernelIndices.emplace_back(kernel);
          // c
          kernelIndices.emplace_back(innerLoops.getInductionVar(cIndex));
          // kX
          for (int i = 0; i < kernelShape.size() - 2; ++i)
            kernelIndices.emplace_back(innerLoops.getInductionVar(i + 1));

          // 4.3 Compute convolution.
          auto loadData =
              rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
          auto loadKernel =
              rewriter.create<LoadOp>(loc, kernelOperand, kernelIndices);
          auto loadPartialSum =
              rewriter.create<LoadOp>(loc, alloc, resultIndices);
          Value result = rewriter.create<AddFOp>(loc, loadPartialSum,
              rewriter.create<MulFOp>(loc, loadData, loadKernel));
          // 4.4 Store computed value into output location.
          rewriter.create<StoreOp>(loc, result, alloc, resultIndices);
        }
      }
    }
    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

struct ONNXConvOpLowering : public ConversionPattern {
  ONNXConvOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConvOp::getOperationName(), 1, ctx) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    ONNXConvOp convOp = llvm::dyn_cast<ONNXConvOp>(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {operands[0]});

    auto resultShape = memRefType.getShape();
    auto &inputOperand = operands[0];
    auto inputShape = inputOperand.getType().cast<MemRefType>().getShape();
    auto &kernelOperand = operands[1];
    auto kernelShape = kernelOperand.getType().cast<MemRefType>().getShape();
    auto &biasOperand = operands[2];
    bool hasBias = !biasOperand.getType().isa<NoneType>();

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
    //
    // kernelsPerGroup = M / group;
    // for n = 0 .. N:
    //   for g = 0 .. group:
    //     for m = 0 .. kernelsPerGroup:
    //       kernel = g * kernelsPerGroup + m;
    //       for r1 = 0 .. RH:
    //         for r2 = 0 .. RW:
    //           R[n][kernel][r1][r2] = 0;
    //           for c = 0 .. C/group:
    //             for k1 = 0 .. KH:
    //               for k2 = 0 .. KW:
    //                 R[n][kernel][r1][r2] =
    //                   D[n][g * (C / group) + c][s1 * r1 + k1][s2 * r2 + k2] *
    //                   K[kernel][c][k1][k2];
    //
    // Naming:
    //   n, g, m: outer loop nest indices
    //   r1, r2: spatial loop nest indices
    //   c, k1, k2: inner loop nest indices
    //
    // TODO: handle padding.
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
    int64_t group = convOp.group().getSExtValue();
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
    outerLoops.createDefineAndOptimizeOp();
    //   for n = 0 .. N:
    int nIndex = outerLoops.pushBounds(0, inputOperand, 0);
    //   for g = 0 .. N:
    int gIndex = -1;
    if (group > 1)
      gIndex = outerLoops.pushBounds(0, group);
    //   for m = 0 .. kernelsPerGroup:
    int mIndex = outerLoops.pushBounds(0, kernelsPerGroup);
    // Outer loop iteration
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
        auto kernelsOffset = rewriter.create<MulIOp>(
            loc, outerLoops.getInductionVar(gIndex), kernelsPerGroupValue);
        kernel = rewriter.create<AddIOp>(
            loc, kernelsOffset, outerLoops.getInductionVar(mIndex));
      }

      // 2.2 Define spatial loops
      int64_t nSpatialLoops = resultShape.size() - 2;
      BuildKrnlLoop spatialLoops(rewriter, loc, nSpatialLoops);
      spatialLoops.createDefineAndOptimizeOp();
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
        rewriter.create<StoreOp>(loc, zero, alloc, resultIndices);

        // 3.2 Define inner loops.
        int64_t nInnerLoops = 1 + (kernelShape.size() - 2);
        BuildKrnlLoop innerLoops(rewriter, loc, nInnerLoops);
        innerLoops.createDefineAndOptimizeOp();
        //   for c = 0 .. C/group
        int cIndex = innerLoops.pushBounds(0, kernelShape[1]);
        //   for Kx = 0 .. KX
        for (int i = 2; i < kernelShape.size(); ++i)
          innerLoops.pushBounds(0, kernelOperand, i);

        // 3.4 Emit inner loop nest.
        innerLoops.createIterateOp();

        // Emit the bias, if needed.
        if (hasBias) {
          auto loadResult =
              rewriter.create<LoadOp>(loc, alloc, resultIndices);
          SmallVector<Value, 4> biasIndices;
          biasIndices.emplace_back(kernel);
          auto loadBias =
              rewriter.create<LoadOp>(loc, biasOperand, kernel);
          auto resultWithBias = rewriter.create<MulFOp>(
            loc, loadResult, loadBias);
          // Store initializer value into output location.
          rewriter.create<StoreOp>(loc, resultWithBias, alloc, resultIndices);
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
          if (group > 1)
            channelDepth = rewriter.create<AddIOp>(loc, channelDepth,
                rewriter.create<MulIOp>(
                    loc, subchannels, outerLoops.getInductionVar(gIndex)));
          dataIndices.emplace_back(channelDepth);
          // sX * rX + kX
          auto stridesAttribute = convOp.stridesAttr();
          // Read strides attribute
          SmallVector<int, 4> strides;
          if (stridesAttribute)
            for (auto stride : stridesAttribute.getValue())
              strides.emplace_back(stride.cast<IntegerAttr>().getInt());
          for (int i = 0; i < kernelShape.size() - 2; ++i) {
            Value spatialIndex = spatialLoops.getInductionVar(i);
            // If strides are present then emit the correct access index.
            if (stridesAttribute && strides[i] > 1)
              spatialIndex = rewriter.create<MulIOp>(loc,
                  rewriter.create<ConstantIndexOp>(loc, strides[i]),
                  spatialLoops.getInductionVar(i));
            dataIndices.emplace_back(rewriter.create<AddIOp>(
                loc, spatialIndex, innerLoops.getInductionVar(i + 1)));
          }

          // 4.2 Prepare indices for accessing the kernel tensor.
          SmallVector<Value, 4> kernelIndices;
          // kernel
          kernelIndices.emplace_back(kernel);
          // c
          kernelIndices.emplace_back(innerLoops.getInductionVar(cIndex));
          // kX
          for (int i = 0; i < kernelShape.size() - 2; ++i)
            kernelIndices.emplace_back(innerLoops.getInductionVar(i + 1));

          // 4.3 Compute convolution.
          auto loadData =
              rewriter.create<LoadOp>(loc, inputOperand, dataIndices);
          auto loadKernel =
              rewriter.create<LoadOp>(loc, kernelOperand, kernelIndices);
          auto loadPartialSum =
              rewriter.create<LoadOp>(loc, alloc, resultIndices);
          Value result = rewriter.create<AddFOp>(loc, loadPartialSum,
              rewriter.create<MulFOp>(loc, loadData, loadKernel));
          // 4.4 Store computed value into output location.
          rewriter.create<StoreOp>(loc, result, alloc, resultIndices);
        }
      }
    }
    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXConvOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLowering>(ctx);
  patterns.insert<ONNXConvNoBiasOpLowering>(ctx);
}
