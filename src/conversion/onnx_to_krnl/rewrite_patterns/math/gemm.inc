//===----- gemm.inc - Lowering Gemm Op ------------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gemm Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

template <typename GemmOp>
struct ONNXGemmOpLowering : public ConversionPattern {
  ONNXGemmOpLowering(MLIRContext *ctx)
      : ConversionPattern(GemmOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    bool hasBias = !op->getOperand(2).getType().isa<NoneType>();

    Value A, B, C;
    A = operands[0];
    B = operands[1];
    if (hasBias)
      C = operands[2];

    auto memRefType = convertToMemRefType(*op->result_type_begin());

    auto alphaAttr =
        FloatAttr::get(memRefType.getElementType(),
                       llvm::dyn_cast<GemmOp>(op).alpha().convertToFloat());
    auto betaAttr =
        FloatAttr::get(memRefType.getElementType(),
                       llvm::dyn_cast<GemmOp>(op).beta().convertToFloat());
    auto alpha = rewriter.create<ConstantOp>(loc, alphaAttr);
    auto beta = rewriter.create<ConstantOp>(loc, betaAttr);

    bool isTransA = (llvm::dyn_cast<GemmOp>(op).transA() != 0);
    bool isTransB = (llvm::dyn_cast<GemmOp>(op).transB() != 0);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      auto memRefShape = memRefType.getShape();
      SmallVector<Value, 2> allocOperands;
      if (memRefShape[0] < 0) {
        auto dim = rewriter.create<DimOp>(loc, A, (isTransA) ? 1 : 0);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[1] < 0) {
        auto dim = rewriter.create<DimOp>(loc, B, (isTransB) ? 0 : 1);
        allocOperands.emplace_back(dim);
      }
      alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t numLoops = 3;

    // Define loops.
    std::vector<Value> originalLoops;
    std::vector<Value> optimizedLoops;
    Block *optimizationBlock =
        defineLoops(rewriter, loc, originalLoops, optimizedLoops, numLoops);

    // We have two Krnl loops:
    // - Outer loop iterates over the output matrix dimensions, and
    // - Reduction loop iterates over the reduction dimension.

    // Outer loop
    std::vector<Value> outerLoops, optimizedOuterLoops;
    outerLoops.reserve(2);
    optimizedOuterLoops.reserve(2);
    for (int i = 0; i < 2; ++i) {
      outerLoops.push_back(originalLoops[i]);
      optimizedOuterLoops.push_back(optimizedLoops[i]);
    }
    KrnlIterateOperandPack outerPack(rewriter, outerLoops, optimizedOuterLoops);
    // Induction variables for the outer loops
    for (int i = 0; i < 2; ++i)
      addDimensionToPack(rewriter, loc, outerPack, alloc, i);

    // Reduction loop
    std::vector<Value> reductionLoops, optimizedReductionLoops;
    reductionLoops.reserve(1);
    optimizedReductionLoops.reserve(1);
    reductionLoops.push_back(originalLoops[2]);
    optimizedReductionLoops.push_back(optimizedLoops[2]);
    KrnlIterateOperandPack reductionPack(rewriter, reductionLoops,
                                         optimizedReductionLoops);
    // Induction variable for the reduction dimension
    // Try to find and use a static value from A or B first.
    // If it failed then use a dynamic value.
    auto ATy = A.getType().cast<MemRefType>();
    auto BTy = B.getType().cast<MemRefType>();
    int64_t K_A_Idx = (isTransA) ? 0 : 1;
    int64_t K_B_Idx = (isTransB) ? 1 : 0;
    reductionPack.pushConstantBound(0);
    if (ATy.getShape()[K_A_Idx] != -1)
      reductionPack.pushConstantBound(ATy.getShape()[K_A_Idx]);
    else if (BTy.getShape()[K_B_Idx] != -1)
      reductionPack.pushConstantBound(BTy.getShape()[K_B_Idx]);
    else
      reductionPack.pushOperandBound(
          rewriter.create<DimOp>(loc, B, K_B_Idx).getResult());

    // Get run-time dimension information for unknown dimensions used for
    // broadcasting.
    // GemmOp supports unidirectional broadcasting from C to A*B.
    // Hence, it must be enough to get broadcasting information for C only.
    std::map<int, Value> broadcastedDimInfo;
    if (hasBias) {
      auto shape = C.getType().cast<MemRefType>().getShape();
      for (int i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
          auto dim = rewriter.create<DimOp>(loc, C, i).getResult();
          auto one = rewriter.create<ConstantIndexOp>(loc, 1);
          auto isBroadcasted =
              rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, dim, one);
          broadcastedDimInfo.insert(std::make_pair(i, isBroadcasted));
        }
      }
    }

    auto outerIterateOp = rewriter.create<KrnlIterateOp>(loc, outerPack);

    // Now perform the insertions into the body of the
    // just generated instructions:

    // No optimization
    rewriter.setInsertionPointToEnd(optimizationBlock);
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);

    // Insert instructions inside the outer loop.
    Block &outerIterationBlock = outerIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&outerIterationBlock);

    // Induction variables
    SmallVector<Value, 4> loopMNIVs;
    for (auto arg : outerIterationBlock.getArguments()) {
      loopMNIVs.emplace_back(arg);
    }

    // Initialize the output of A*B
    auto zero = rewriter.create<ConstantOp>(
        loc, FloatAttr::get(memRefType.getElementType(), 0));
    rewriter.create<StoreOp>(loc, zero, alloc, loopMNIVs);

    // Compute A*B
    auto matmulIterateOp = rewriter.create<KrnlIterateOp>(loc, reductionPack);

    // Compute beta*C, and add up to alpha*A*B (unidirectional broadcasting)
    auto loadedAB = rewriter.create<LoadOp>(loc, alloc, loopMNIVs);
    auto alphaAB = rewriter.create<MulFOp>(loc, alpha, loadedAB);
    if (hasBias) {
      auto loopCIVs = getLoopIVsForBroadcasting(loc, rewriter, loopMNIVs, C,
                                                broadcastedDimInfo);
      auto loadedC = rewriter.create<LoadOp>(loc, C, loopCIVs);
      auto betaC = rewriter.create<MulFOp>(loc, beta, loadedC);
      auto Y = rewriter.create<AddFOp>(loc, alphaAB, betaC);
      rewriter.create<StoreOp>(loc, Y, alloc, loopMNIVs);
    } else {
      rewriter.create<StoreOp>(loc, alphaAB, alloc, loopMNIVs);
    }

    // Insert instructions to do matrix multiplication: A*B
    Block &matmulIterationBlock = matmulIterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&matmulIterationBlock);

    // Induction variables
    SmallVector<Value, 4> loopKIVs, loopAIVs, loopBIVs;
    for (auto arg : matmulIterationBlock.getArguments())
      loopKIVs.emplace_back(arg);
    if (isTransA) {
      loopAIVs.emplace_back(loopKIVs[0]);
      loopAIVs.emplace_back(loopMNIVs[0]);
    } else {
      loopAIVs.emplace_back(loopMNIVs[0]);
      loopAIVs.emplace_back(loopKIVs[0]);
    }
    if (isTransB) {
      loopBIVs.emplace_back(loopMNIVs[1]);
      loopBIVs.emplace_back(loopKIVs[0]);
    } else {
      loopBIVs.emplace_back(loopKIVs[0]);
      loopBIVs.emplace_back(loopMNIVs[1]);
    }

    // Matmul computation
    auto loadedA = rewriter.create<LoadOp>(loc, A, loopAIVs);
    auto loadedB = rewriter.create<LoadOp>(loc, B, loopBIVs);
    auto loadedY = rewriter.create<LoadOp>(loc, alloc, loopMNIVs);
    auto AB = rewriter.create<MulFOp>(loc, loadedA, loadedB);
    auto accumulated = rewriter.create<AddFOp>(loc, loadedY, AB);
    rewriter.create<StoreOp>(loc, accumulated, alloc, loopMNIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXGemmOpPattern(OwningRewritePatternList &patterns,
                                       MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLowering<ONNXGemmOp>>(ctx);
}
