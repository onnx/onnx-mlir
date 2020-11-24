#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXInstanceNormalizationOpLowering : public ConversionPattern {
  ONNXInstanceNormalizationOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXInstanceNormalizationOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // instancenorm{epsilon}(x, scale, bias) =
    //      scale * (x - mean) / sqrt(variance + epsilon) + bias
    ONNXInstanceNormalizationOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto epsilonAttr = FloatAttr::get(memRefType.getElementType(),
        llvm::dyn_cast<ONNXInstanceNormalizationOp>(op)
            .epsilon()
            .convertToFloat());
    auto epsilon = rewriter.create<ConstantOp>(loc, epsilonAttr);

    auto operand = operandAdaptor.input();
    auto scale = operandAdaptor.scale();
    auto bias = operandAdaptor.B();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(
          memRefType, loc, rewriter, insertDealloc, {operand});

    // Operand's dimensions can be in the form of NxCxD1xD2x...xDn or N.
    // In case of N, C is assumed to be 1.
    // Shapes of scale, bias, mean and variance must be C.
    // Computation of BatchNormalization is done as if scale, bias, mean, and
    // variance are reshaped to Cx1x1x...x1.

    // rank
    int64_t rank = memRefType.getRank();

    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, rank);

    // Create a KrnlIterateOp along C dimension.
    // This will be the outer-most loop in order to re-use scale, bias

    SmallVector<Value, 1> loopCIVs;
    if (rank > 1) {
      KrnlIterateOperandPack cPack(rewriter, originalLoops[1]);
      addDimensionToPack(rewriter, loc, cPack, operand, 1);
      auto cIterateOp = rewriter.create<KrnlIterateOp>(loc, cPack);
      Block &cIterationBlock = cIterateOp.bodyRegion().front();
      rewriter.setInsertionPointToStart(&cIterationBlock);
      for (auto arg : cIterationBlock.getArguments())
        loopCIVs.emplace_back(arg);
    } else {
      loopCIVs.emplace_back(rewriter.create<ConstantIndexOp>(loc, 0));
    }

    auto scaleVal = rewriter.create<AffineLoadOp>(loc, scale, loopCIVs);
    auto biasVal = rewriter.create<AffineLoadOp>(loc, bias, loopCIVs);

    // Create a KrnlIterateOp along the other dimensions.
    SmallVector<int64_t, 4> axes;
    axes.emplace_back(0);
    for (int64_t i = 2; i < rank; ++i)
      axes.emplace_back(i);
    std::vector<Value> packLoops;
    for (int i = 0; i < axes.size(); ++i) {
      packLoops.emplace_back(originalLoops[axes[i]]);
    }
    KrnlIterateOperandPack pack(rewriter, packLoops);
    for (int i = 0; i < axes.size(); ++i) {
      addDimensionToPack(rewriter, loc, pack, operand, axes[i]);
    }
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);

    Block &iterationBlock = iterateOp.bodyRegion().front();
    rewriter.setInsertionPointToStart(&iterationBlock);

    SmallVector<Value, 4> loopIVs;
    auto args = iterationBlock.getArguments();
    if (args.size() > 1) {
      loopIVs.emplace_back(args[0]);
      loopIVs.emplace_back(loopCIVs[0]); // Insert C back.
      for (int i = 1; i < args.size(); ++i)
        loopIVs.emplace_back(args[i]);
    } else if (rank == 2) {
      loopIVs.emplace_back(args[0]);
      loopIVs.emplace_back(loopCIVs[0]); // Insert C back.
    } else {
      loopIVs.emplace_back(args[0]);
    }

    auto xVal = rewriter.create<AffineLoadOp>(loc, operand, loopIVs);

    // Get mean and var
    auto inputShapeAttr = rewriter.getF16FloatAttr(
        (float)operand.getType().cast<RankedTensorType>().getShape().size());
    auto inputShape = rewriter.create<ConstantOp>(loc, inputShapeAttr);

    Type reduced;
    auto meanVal = rewriter.create<ONNXReduceMeanOp>(loc, reduced, operand);

    auto varNumerator = rewriter.create<SubFOp>(loc, xVal, meanVal);
    auto varNumeratorSqr =
        rewriter.create<MulFOp>(loc, varNumerator, varNumerator);
    auto varianceVal =
        rewriter.create<DivFOp>(loc, varNumeratorSqr, inputShape);

    // Normalize
    auto dividend = rewriter.create<SubFOp>(loc, xVal, xVal);
    auto adjustedVarianceVal =
        rewriter.create<AddFOp>(loc, varianceVal, epsilon);
    auto divisor = rewriter.create<SqrtOp>(loc, adjustedVarianceVal);
    auto normVal = rewriter.create<DivFOp>(loc, dividend, divisor);

    // Scale and shift
    auto scaleNormVal = rewriter.create<MulFOp>(loc, scaleVal, normVal);
    auto shiftScaleNormVal =
        rewriter.create<AddFOp>(loc, scaleNormVal, biasVal);
    rewriter.create<AffineStoreOp>(loc, shiftScaleNormVal, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXInstanceNormalizationOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXInstanceNormalizationOpLowering>(ctx);
}
