#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXMeanVarianceNormalizationOpLowering
    : public OpConversionPattern<ONNXMeanVarianceNormalizationOp> {
  ONNXMeanVarianceNormalizationOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXMeanVarianceNormalizationOp op,
      ONNXMeanVarianceNormalizationOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = ONNXLoc<ONNXMeanVarianceNormalizationOp>(op);
    Value X = adaptor.getX();
    ArrayAttr axesAttr = adaptor.getAxes();

    MultiDialectBuilder<MathBuilder, MemRefBuilder, KrnlBuilder,
        IndexExprBuilderForKrnl>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);
    MemRefType XType = X.getType().cast<MemRefType>();
    Type elementType = XType.getElementType();
    int64_t rank = XType.getRank();

    SmallVector<int64_t> axes;
    for (Attribute attr : axesAttr)
      axes.push_back(cast<IntegerAttr>(attr).getInt());

    // Compute divisor (product of sizes along reduction axes)
    // This divisor is the number of elements being reduced over:
    float div = 1.0;
    ArrayRef<int64_t> inputShape = XType.getShape();
    for (int64_t axis : axes)
      div *= inputShape[axis];
    Value divValue = create.math.constant(elementType, div);

    // Create reduction shape:
    // shape with 1 on axes being reduced, inputShape[i] elsewhere
    SmallVector<int64_t> reductionShape;
    for (int64_t i = 0; i < rank; ++i)
      reductionShape.push_back(llvm::is_contained(axes, i) ? 1 : inputShape[i]);
    MemRefType reductionType = MemRefType::get(reductionShape, elementType);

    // Initialize accumulators for sum(X) and sum(X^2)
    Value zero = create.math.constant(elementType, 0.0);
    Value sumX = create.mem.alignedAlloc(reductionType);
    Value sumX2 = create.mem.alignedAlloc(reductionType);
    create.krnl.memset(sumX, zero);
    create.krnl.memset(sumX2, zero);

    // Define full input loop bounds
    SmallVector<IndexExpr> lbs(rank, LitIE(0)), ubs;
    for (int64_t i = 0; i < rank; ++i)
      ubs.emplace_back(create.krnlIE.getShapeAsDim(X, i));

    // Loop over full input tensor to accumulate sum and sum of squares:
    // sumX = ∑ X
    // sumX2 = ∑ X^2
    ValueRange loops = create.krnl.defineLoops(rank);
    create.krnl.iterateIE(loops, loops, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopIVs) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
          Value x = create.krnl.load(X, loopIVs);
          Value x2 = create.math.mul(x, x);

          // Map full indices to reduction indices by replacing
          // axes indices with 0 (since reduced dims collapse)
          SmallVector<Value> redIVs;
          for (int64_t i = 0; i < rank; ++i)
            redIVs.push_back(llvm::is_contained(axes, i)
                                 ? create.math.constantIndex(0)
                                 : loopIVs[i]);

          Value prevSum = create.krnl.load(sumX, redIVs);
          Value prevSum2 = create.krnl.load(sumX2, redIVs);
          create.krnl.store(create.math.add(prevSum, x), sumX, redIVs);
          create.krnl.store(create.math.add(prevSum2, x2), sumX2, redIVs);
        });

    // Allocate buffers for mean and standard deviation:
    Value mean = create.mem.alignedAlloc(reductionType);
    Value stddev = create.mem.alignedAlloc(reductionType);

    // Loop over reduced shape to compute mean and stddev:
    // mean = sumX / divisor
    // variance = (sumX2 / divisor) - mean^2
    // stddev = sqrt(abs(variance))
    SmallVector<IndexExpr> redLbs(rank, LitIE(0)), redUbs;
    for (int64_t i = 0; i < rank; ++i)
      redUbs.emplace_back(llvm::is_contained(axes, i)
                              ? IndexExpr(LitIE(1))
                              : create.krnlIE.getShapeAsDim(X, i));

    ValueRange redLoops = create.krnl.defineLoops(rank);
    create.krnl.iterateIE(redLoops, redLoops, redLbs, redUbs,
        [&](const KrnlBuilder &createKrnl, ValueRange redIVs) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);

          Value sX = create.krnl.load(sumX, redIVs);
          Value sX2 = create.krnl.load(sumX2, redIVs);
          Value m = create.math.div(sX, divValue);
          Value var = create.math.div(sX2, divValue);

          Value sub = create.math.sub(var, create.math.mul(m, m));
          // Variance can be slightly negative due to numerical error, so abs is
          // used
          Value std = create.math.sqrt(create.math.abs(sub));

          create.krnl.store(m, mean, redIVs);
          create.krnl.store(std, stddev, redIVs);
        });

    // Allocate buffer for normalized output tensor (same shape as X)
    Value norm = create.mem.alignedAlloc(XType);

    // Normalize input tensor:
    // For each element x:
    //   normalized_x = (x - mean) / stddev
    // where mean and stddev are broadcast over the reduced axes.
    ValueRange normLoops = create.krnl.defineLoops(rank);
    create.krnl.iterateIE(normLoops, normLoops, lbs, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopIVs) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);

          SmallVector<Value> redIVs;
          for (int64_t i = 0; i < rank; ++i)
            redIVs.push_back(llvm::is_contained(axes, i)
                                 ? create.math.constantIndex(0)
                                 : loopIVs[i]);

          Value x = create.krnl.load(X, loopIVs);
          Value m = create.krnl.load(mean, redIVs);
          Value std = create.krnl.load(stddev, redIVs);
          Value diff = create.math.sub(x, m);
          Value normVal = create.math.div(diff, std);
          create.krnl.store(normVal, norm, loopIVs);
        });

    rewriter.replaceOp(op, norm);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXMeanVarianceNormalizationOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMeanVarianceNormalizationOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
