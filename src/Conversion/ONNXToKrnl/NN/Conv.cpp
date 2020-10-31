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

struct ONNXConvOpLowering : public ConversionPattern {
  ONNXConvOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXConvOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ONNXConvOpAdaptor operandAdaptor(operands);
    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    ONNXConvOp convOp = llvm::dyn_cast<ONNXConvOp>(op);

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

        // 3.2 Define inner loops.
        int64_t nInnerLoops = 1 + (kernelShape.size() - 2);
        BuildKrnlLoop innerLoops(rewriter, loc, nInnerLoops);
        innerLoops.createDefineOp();
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
            int stride = 1;
            if (stridesAttribute && strides[i] > 1)
              stride = strides[i];
            AffineMap indexMap = AffineMap::get(2, 0,
                /*sX=*/rewriter.getAffineDimExpr(0) * /*rX=*/stride +
                    /*kX=*/rewriter.getAffineDimExpr(1));
            Value outIV = rewriter.create<AffineApplyOp>(loc, indexMap,
                ArrayRef<Value>{spatialLoops.getInductionVar(i),
                    innerLoops.getInductionVar(i + 1)});
            dataIndices.emplace_back(outIV);
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
              rewriter.create<AffineLoadOp>(loc, inputOperand, dataIndices);
          auto loadKernel =
              rewriter.create<AffineLoadOp>(loc, kernelOperand, kernelIndices);
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

struct ONNXLoopOpLowering : public ConversionPattern {
  ONNXLoopOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXLoopOp::getOperationName(), 1, ctx) {}

  void allocateMemoryForVFinal(mlir::Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXLoopOpAdaptor loopOpAdapter,
      SmallVectorImpl<mlir::Value> &outputs) const {
    auto vFinalAndScanOutputs = op->getOpResults();
    auto opVFinalOutputs = llvm::make_range(vFinalAndScanOutputs.begin(),
        vFinalAndScanOutputs.begin() + loopOpAdapter.v_initial().size());
    auto vInitIter = loopOpAdapter.v_initial();

    for (const auto &ioPair : llvm::zip(vInitIter, opVFinalOutputs)) {
      auto vInit = std::get<0>(ioPair);
      auto vFinal = std::get<1>(ioPair);

      auto memRefType = convertToMemRefType(vFinal.getType());
      Value alloc;
      bool shouldDealloc = checkInsertDealloc(op);
      if (hasAllConstantDimensions(memRefType))
        alloc = insertAllocAndDealloc(memRefType, loc, rewriter, shouldDealloc);
      else
        alloc = insertAllocAndDealloc(
            memRefType, loc, rewriter, shouldDealloc, {vInit});
      outputs.emplace_back(alloc);
    }
  }

  void allocateMemoryForScanOutput(mlir::Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXLoopOpAdaptor loopOpAdapter,
      SmallVectorImpl<mlir::Value> &outputs) const {
    auto vFinalAndScanOutputs = op->getOpResults();
    auto opScanOutputIter = llvm::make_range(
        vFinalAndScanOutputs.begin() + loopOpAdapter.v_initial().size(),
        vFinalAndScanOutputs.end());
    auto vInitIter = loopOpAdapter.v_initial();

    // Are the correspondence guaranteed?
    for (const auto &ioPair : llvm::zip(vInitIter, opScanOutputIter)) {
      auto vInit = std::get<0>(ioPair);
      auto opScanOutput = std::get<1>(ioPair);

      auto memRefType = convertToMemRefType(opScanOutput.getType());
      Value alloc;
      bool shouldDealloc = checkInsertDealloc(op);
      if (hasAllConstantDimensions(memRefType))
        alloc = insertAllocAndDealloc(memRefType, loc, rewriter, shouldDealloc);
      else {
        auto rankedScanOutTy = memRefType;
        SmallVector<mlir::Value, 4> allocParams;

        for (int i = 0; i < rankedScanOutTy.getRank(); i++) {
          if (rankedScanOutTy.getShape()[i] == -1) {
            if (i == 0) {
              // TODO(tjingrant): in general, it is not correct to expect
              // loop operation scan output to have the leading dimension extent
              // equal to the trip count, due to the possibility of early
              // termination.
              assert(!loopOpAdapter.M().getType().isa<NoneType>());
              Value maxTripCount =
                  rewriter.create<LoadOp>(loc, loopOpAdapter.M()).getResult();
              allocParams.emplace_back(rewriter.create<IndexCastOp>(
                  loc, maxTripCount, rewriter.getIndexType()));
            } else {
              //              allocParams.emplace_back(
              //                  rewriter.create<DimOp>(loc, vInit, i -
              //                  1).getResult());
              llvm_unreachable("Error.");
            }
          }
        }
        alloc = rewriter.create<AllocOp>(loc, rankedScanOutTy, allocParams);
      }
      outputs.emplace_back(alloc);
    }
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ONNXLoopOpAdaptor loopOpAdapter(operands, op->getAttrDictionary());

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto symbolName =
        loopOpAdapter.body().cast<SymbolRefAttr>().getLeafReference();
    auto func = dyn_cast<mlir::FuncOp>(module.lookupSymbol(symbolName));
    auto &loopBody = func.getBody();

    // Allocate memory for two kinds of outputs:
    // - final values of loop dependent variables, and
    // - scan output (all intermediate values of) loop dependent variables.
    SmallVector<Value, 4> outputs;
    allocateMemoryForVFinal(loc, rewriter, op, loopOpAdapter, outputs);
    allocateMemoryForScanOutput(loc, rewriter, op, loopOpAdapter, outputs);

    // Copy content of vInit to vFinal, which is used to host intermediate
    // values produced by loop body function invocation.
    for (const auto &vInitAndFinal :
        llvm::zip(loopOpAdapter.v_initial(), outputs)) {
      const auto &vInit = std::get<0>(vInitAndFinal);
      const auto &vFinal = std::get<1>(vInitAndFinal);
      EmitCopy(rewriter, loc, vInit, vFinal);
    }

    BuildKrnlLoop loop(rewriter, loc, 1);
    loop.createDefineOp();
    Value maxTripCount =
        rewriter.create<LoadOp>(loc, loopOpAdapter.M()).getResult();
    maxTripCount = rewriter.create<IndexCastOp>(
        loc, maxTripCount, rewriter.getIndexType());
    loop.pushBounds(0, maxTripCount);
    loop.createIterateOp();
    rewriter.setInsertionPointToStart(loop.getIterateBlock());

    // Create a scalar tensor out of iv, as the first argument passed to the
    // body graph function.
    Value iv = loop.getInductionVar(0);
    iv = rewriter.create<IndexCastOp>(loc, iv, rewriter.getI64Type())
             .getResult();
    Value ivMemRef =
        rewriter
            .create<AllocOp>(loc, MemRefType::get({}, rewriter.getI64Type()))
            .getResult();
    rewriter.create<StoreOp>(loc, iv, ivMemRef);

    // Make the call to loop body function.
    SmallVector<Value, 4> params = {ivMemRef, loopOpAdapter.cond()};
    for (auto value : llvm::make_range(outputs.begin(),
             outputs.begin() + loopOpAdapter.v_initial().size()))
      params.emplace_back(value);

    auto callOp = rewriter.create<CallOp>(loc, func, params);

    // Post values from loop body function.
    auto resultsRange = callOp.getResults();
    SmallVector<Value, 4> bodyOutputs(resultsRange.begin(), resultsRange.end());

    for (int i = 0; i < bodyOutputs.size(); i++) {
      auto output = bodyOutputs[i];
      assert(output.getType().isa<TensorType>() ||
             output.getType().isa<MemRefType>() &&
                 "Expecting loop body function output to consist of "
                 "tensors/memrefs.");
      auto outputTy = output.getType().cast<ShapedType>();
      bodyOutputs[i] = rewriter
                           .create<KrnlDummyCastOp>(loc, output,
                               MemRefType::get(outputTy.getShape(),
                                   outputTy.getElementType()))
                           .getResult();
    }

    auto vIntermediate = llvm::make_range(bodyOutputs.begin() + 1,
        bodyOutputs.begin() + 1 + loopOpAdapter.v_initial().size());
    for (auto vIntermediateToFinal : llvm::zip(vIntermediate, outputs))
      EmitCopy(rewriter, loc, std::get<0>(vIntermediateToFinal),
          std::get<1>(vIntermediateToFinal));

    rewriter.replaceOp(op, outputs);
    return success();
  }

  void EmitCopy(ConversionPatternRewriter &rewriter, const Location &loc,
      const Value &vInit, const Value &vFinal) const {
    OpBuilder::InsertionGuard insertGuard(rewriter);
    auto vInitTy = vInit.getType().cast<MemRefType>();
    BuildKrnlLoop loop(rewriter, loc, vInitTy.getRank());
    loop.createDefineOp();
    for (int i = 0; i < vInitTy.getRank(); i++)
      loop.pushBounds(0, vInit, i);
    loop.createIterateOp();
    rewriter.setInsertionPointToStart(loop.getIterateBlock());
    auto allIV = loop.getAllInductionVar();
    auto v = rewriter.create<AffineLoadOp>(loc, vInit, allIV).getResult();
    rewriter.create<AffineStoreOp>(loc, v, vFinal, allIV);
  }
};

void populateLoweringONNXConvOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLowering>(ctx);
  patterns.insert<ONNXLoopOpLowering>(ctx);
}
