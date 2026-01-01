/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
//
// Pattern to convert ONNX operations to their ChannelLast variants.
//
// This pass wraps channel-first (NCHW) operations with transpose operations
// to enable the use of channel-last (NHWC) kernels. The pattern:
// 1. Transposes input from NCHW to channel-last layout
// 2. Applies the ChannelLast operation
// 3. Transposes output back to NCHW layout
//
// Supported conversions:
// - Conv -> XFEConv
// - ConvTranspose -> XFEConvTranspose
// - AveragePool -> XFEAveragePool
// - MaxPool -> XFEMaxPool
// - GlobalAveragePool -> XFEGlobalAveragePool
// - GlobalMaxPool -> XFEGlobalMaxPool
// - InstanceNormalization -> XFEInstanceNormalization
// - DepthToSpace -> XFEDepthToSpace
// - SpaceToDepth -> XFESpaceToDepth
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

// Helper function to create input transpose (NCHW -> channel-last)
// For rank N: [0, 2, 3, ..., N-1, 1]
Value createInputTranspose(PatternRewriter &rewriter, Location loc, Value input,
    int64_t rank, Type elementType) {
  SmallVector<int64_t, 4> perm;
  perm.push_back(0); // batch
  for (int64_t i = 2; i < rank; ++i)
    perm.push_back(i); // spatial dimensions
  perm.push_back(1);   // channels

  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  SmallVector<int64_t, 4> transposedShape;
  for (int64_t p : perm) {
    transposedShape.push_back(inputType.getDimSize(p));
  }

  auto transposedType =
      RankedTensorType::get(transposedShape, inputType.getElementType());
  return rewriter.create<ONNXTransposeOp>(
      loc, transposedType, input, rewriter.getI64ArrayAttr(perm));
}

// Helper function to create weight transpose for Conv
// Conv: OIHW -> OHWI (permutation [0, 2, 3, ..., N-1, 1])
// Keeps output channels first, moves input channels to last
Value createWeightTranspose(PatternRewriter &rewriter, Location loc,
    Value weight, int64_t rank, Type elementType) {
  SmallVector<int64_t, 4> perm;
  perm.push_back(0); // output channels
  for (int64_t i = 2; i < rank; ++i)
    perm.push_back(i); // spatial dimensions (H, W)
  perm.push_back(1);   // input channels

  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  SmallVector<int64_t, 4> transposedShape;
  for (int64_t p : perm) {
    transposedShape.push_back(weightType.getDimSize(p));
  }

  auto transposedType =
      RankedTensorType::get(transposedShape, weightType.getElementType());
  return rewriter.create<ONNXTransposeOp>(
      loc, transposedType, weight, rewriter.getI64ArrayAttr(perm));
}

// Helper function to create weight transpose for ConvTranspose
// ConvTranspose: IOHW -> OHWI (permutation [1, 2, 3, ..., N-1, 0])
// Swaps I and O, keeps spatial dimensions, moves I to last
Value createConvTransposeWeightTranspose(PatternRewriter &rewriter,
    Location loc, Value weight, int64_t rank, Type elementType) {
  SmallVector<int64_t, 4> perm;
  perm.push_back(1); // output channels (was position 1 in IOHW)
  for (int64_t i = 2; i < rank; ++i)
    perm.push_back(i); // spatial dimensions (H, W)
  perm.push_back(0);   // input channels (was position 0 in IOHW)

  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  SmallVector<int64_t, 4> transposedShape;
  for (int64_t p : perm) {
    transposedShape.push_back(weightType.getDimSize(p));
  }

  auto transposedType =
      RankedTensorType::get(transposedShape, weightType.getElementType());
  return rewriter.create<ONNXTransposeOp>(
      loc, transposedType, weight, rewriter.getI64ArrayAttr(perm));
}

// Helper function to create output transpose (channel-last -> NCHW)
// For rank N: [0, N-1, 1, 2, ..., N-2]
Value createOutputTranspose(PatternRewriter &rewriter, Location loc,
    Value output, Type outputType, int64_t rank) {
  SmallVector<int64_t, 4> perm;
  perm.push_back(0);        // batch
  perm.push_back(rank - 1); // channels (last dimension)
  for (int64_t i = 1; i < rank - 1; ++i)
    perm.push_back(i); // spatial dimensions

  return rewriter.create<ONNXTransposeOp>(
      loc, outputType, output, rewriter.getI64ArrayAttr(perm));
}

// Pattern to convert Conv to XFEConv
struct ConvToChannelLastPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern<ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();
    Value input = convOp.getX();
    Value weight = convOp.getW();
    Value bias = convOp.getB();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    auto weightType = mlir::dyn_cast<RankedTensorType>(weight.getType());

    if (!inputType || !weightType)
      return failure();

    // Support N-dimensional tensors (rank >= 3)
    if (inputType.getRank() < 3 || weightType.getRank() < 3)
      return failure();

    int64_t rank = inputType.getRank();

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Transpose weight to channel-last
    Value weightChannelLast = createWeightTranspose(
        rewriter, loc, weight, rank, weightType.getElementType());

    // Create XFEConv operation
    // CRITICAL: Use original Conv's output element type to preserve
    // quantization
    auto origOutputType = mlir::cast<ShapedType>(convOp.getType());
    Type outputElementType = origOutputType.getElementType();
    auto convChannelLastOp = rewriter.create<XFEConvOp>(loc,
        UnrankedTensorType::get(outputElementType), inputChannelLast,
        weightChannelLast, bias, convOp.getAutoPadAttr(),
        convOp.getDilationsAttr(), convOp.getGroupAttr(),
        convOp.getKernelShapeAttr(), convOp.getPadsAttr(),
        convOp.getStridesAttr());
    
    // CRITICAL: Immediately run shape inference to resolve unranked type
    // This ensures the output has correct shape AND element type before
    // creating Transpose
    if (failed(convChannelLastOp.inferShapes(nullptr))) {
      return failure();
    }

    // Transpose output back to NCHW
    // CRITICAL: Get element type from XFEConv's ACTUAL output (after shape
    // inference). This ensures we preserve quantized types that were set during
    // shape inference
    auto xfeOutputType =
        mlir::cast<ShapedType>(convChannelLastOp.getResult().getType());
    Type actualElementType = xfeOutputType.getElementType();
    Type transposeOutputType;
    if (origOutputType.hasRank()) {
      // Use original Conv's shape but XFEConv's actual element type
      transposeOutputType = RankedTensorType::get(
          origOutputType.getShape(), actualElementType);
    } else {
      transposeOutputType = convOp.getType();
    }

    Value outputNCHW =
        createOutputTranspose(rewriter, loc, convChannelLastOp.getResult(),
                              transposeOutputType, rank);

    rewriter.replaceOp(convOp, outputNCHW);
    return success();
  }
};

// Pattern to convert ConvTranspose to XFEConvTranspose
struct ConvTransposeToChannelLastPattern
    : public OpRewritePattern<ONNXConvTransposeOp> {
  using OpRewritePattern<ONNXConvTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXConvTransposeOp convTransposeOp,
      PatternRewriter &rewriter) const override {
    Location loc = convTransposeOp.getLoc();
    Value input = convTransposeOp.getX();
    Value weight = convTransposeOp.getW();
    Value bias = convTransposeOp.getB();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    auto weightType = mlir::dyn_cast<RankedTensorType>(weight.getType());

    if (!inputType || !weightType)
      return failure();

    // Support N-dimensional tensors (rank >= 3)
    if (inputType.getRank() < 3 || weightType.getRank() < 3)
      return failure();

    int64_t rank = inputType.getRank();

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Transpose weight to channel-last (ConvTranspose uses IOHW layout -> OHWI)
    Value weightChannelLast = createConvTransposeWeightTranspose(
        rewriter, loc, weight, rank, weightType.getElementType());

    // Create XFEConvTranspose operation
    // CRITICAL: Use original ConvTranspose's output element type to preserve
    // quantization
    auto origOutputType = mlir::cast<ShapedType>(convTransposeOp.getType());
    Type outputElementType = origOutputType.getElementType();
    auto convTransposeChannelLastOp = rewriter.create<XFEConvTransposeOp>(loc,
        UnrankedTensorType::get(outputElementType), inputChannelLast,
        weightChannelLast, bias, convTransposeOp.getAutoPadAttr(),
        convTransposeOp.getDilationsAttr(), convTransposeOp.getGroupAttr(),
        convTransposeOp.getKernelShapeAttr(),
        convTransposeOp.getOutputPaddingAttr(),
        convTransposeOp.getOutputShapeAttr(), convTransposeOp.getPadsAttr(),
        convTransposeOp.getStridesAttr());
    
    // CRITICAL: Immediately run shape inference
    if (failed(convTransposeChannelLastOp.inferShapes(nullptr))) {
      return failure();
    }

    // Transpose output back to NCHW
    // CRITICAL: Get element type from XFEConvTranspose's ACTUAL output (after
    // shape inference)
    auto xfeOutputType = mlir::cast<ShapedType>(
        convTransposeChannelLastOp.getResult().getType());
    Type actualElementType = xfeOutputType.getElementType();
    Type transposeOutputType;
    if (origOutputType.hasRank()) {
      // Use original ConvTranspose's shape but XFEConvTranspose's actual
      // element type
      transposeOutputType = RankedTensorType::get(
          origOutputType.getShape(), actualElementType);
    } else {
      transposeOutputType = convTransposeOp.getType();
    }
    
    Value outputNCHW = createOutputTranspose(rewriter, loc,
        convTransposeChannelLastOp.getResult(), transposeOutputType,
        rank);

    rewriter.replaceOp(convTransposeOp, outputNCHW);
    return success();
  }
};

// Pattern to convert AveragePool to XFEAveragePool
struct AveragePoolToChannelLastPattern
    : public OpRewritePattern<ONNXAveragePoolOp> {
  using OpRewritePattern<ONNXAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAveragePoolOp poolOp, PatternRewriter &rewriter) const override {
    Location loc = poolOp.getLoc();
    Value input = poolOp.getX();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() < 3)
      return failure();

    int64_t rank = inputType.getRank();

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Create XFEAveragePool operation
    // Use original pool's output element type to preserve quantization
    auto origOutputType = mlir::cast<ShapedType>(poolOp.getType());
    Type outputElementType = origOutputType.getElementType();
    auto poolChannelLastOp = rewriter.create<XFEAveragePoolOp>(loc,
        UnrankedTensorType::get(outputElementType), inputChannelLast,
        poolOp.getAutoPadAttr(), poolOp.getCeilModeAttr(),
        poolOp.getCountIncludePadAttr(), poolOp.getKernelShapeAttr(),
        poolOp.getPadsAttr(), poolOp.getStridesAttr());
    
    // CRITICAL: Immediately run shape inference
    if (failed(poolChannelLastOp.inferShapes(nullptr))) {
      return failure();
    }

    // Transpose output back to NCHW
    // Get actual element type from XFEAveragePool output
    auto xfeOutputType =
        mlir::cast<ShapedType>(poolChannelLastOp.getResult().getType());
    auto transposeOutputType = RankedTensorType::get(
        origOutputType.getShape(), xfeOutputType.getElementType());
    Value outputNCHW =
        createOutputTranspose(rewriter, loc, poolChannelLastOp.getResult(),
                              transposeOutputType, rank);

    rewriter.replaceOp(poolOp, outputNCHW);
    return success();
  }
};

// Pattern to convert MaxPool to XFEMaxPool
struct MaxPoolToChannelLastPattern
    : public OpRewritePattern<ONNXMaxPoolSingleOutOp> {
  using OpRewritePattern<ONNXMaxPoolSingleOutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMaxPoolSingleOutOp poolOp, PatternRewriter &rewriter) const override {
    Location loc = poolOp.getLoc();
    Value input = poolOp.getX();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() < 3)
      return failure();

    int64_t rank = inputType.getRank();

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Create XFEMaxPool operation
    // Use original pool's output element type to preserve quantization
    auto origOutputType = mlir::cast<ShapedType>(poolOp.getType());
    Type outputElementType = origOutputType.getElementType();
    auto poolChannelLastOp = rewriter.create<XFEMaxPoolOp>(loc,
        UnrankedTensorType::get(outputElementType), inputChannelLast,
        poolOp.getAutoPadAttr(), poolOp.getCeilModeAttr(),
        poolOp.getDilationsAttr(), poolOp.getKernelShapeAttr(),
        poolOp.getPadsAttr(), poolOp.getStorageOrderAttr(),
        poolOp.getStridesAttr());
    
    // CRITICAL: Immediately run shape inference
    if (failed(poolChannelLastOp.inferShapes(nullptr))) {
      return failure();
    }

    // Transpose output back to NCHW
    // Get actual element type from XFEMaxPool output
    auto xfeOutputType =
        mlir::cast<ShapedType>(poolChannelLastOp.getResult().getType());
    auto transposeOutputType = RankedTensorType::get(
        origOutputType.getShape(), xfeOutputType.getElementType());
    Value outputNCHW =
        createOutputTranspose(rewriter, loc, poolChannelLastOp.getResult(),
                              transposeOutputType, rank);

    rewriter.replaceOp(poolOp, outputNCHW);
    return success();
  }
};

// Pattern to convert GlobalAveragePool to XFEGlobalAveragePool
struct GlobalAveragePoolToChannelLastPattern
    : public OpRewritePattern<ONNXGlobalAveragePoolOp> {
  using OpRewritePattern<ONNXGlobalAveragePoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXGlobalAveragePoolOp poolOp,
      PatternRewriter &rewriter) const override {
    Location loc = poolOp.getLoc();
    Value input = poolOp.getX();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() < 3)
      return failure();

    int64_t rank = inputType.getRank();

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Create XFEGlobalAveragePool operation
    auto poolChannelLastOp = rewriter.create<XFEGlobalAveragePoolOp>(loc,
        UnrankedTensorType::get(inputType.getElementType()), inputChannelLast);

    // Transpose output back to NCHW
    Value outputNCHW = createOutputTranspose(
        rewriter, loc, poolChannelLastOp.getResult(), poolOp.getType(), rank);

    rewriter.replaceOp(poolOp, outputNCHW);
    return success();
  }
};

// Pattern to convert GlobalMaxPool to XFEGlobalMaxPool
struct GlobalMaxPoolToChannelLastPattern
    : public OpRewritePattern<ONNXGlobalMaxPoolOp> {
  using OpRewritePattern<ONNXGlobalMaxPoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXGlobalMaxPoolOp poolOp, PatternRewriter &rewriter) const override {
    Location loc = poolOp.getLoc();
    Value input = poolOp.getX();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() < 3)
      return failure();

    int64_t rank = inputType.getRank();

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Create XFEGlobalMaxPool operation
    auto poolChannelLastOp = rewriter.create<XFEGlobalMaxPoolOp>(loc,
        UnrankedTensorType::get(inputType.getElementType()), inputChannelLast);

    // Transpose output back to NCHW
    Value outputNCHW = createOutputTranspose(
        rewriter, loc, poolChannelLastOp.getResult(), poolOp.getType(), rank);

    rewriter.replaceOp(poolOp, outputNCHW);
    return success();
  }
};

// Pattern to convert InstanceNormalization to XFEInstanceNormalization
struct InstanceNormToChannelLastPattern
    : public OpRewritePattern<ONNXInstanceNormalizationOp> {
  using OpRewritePattern<ONNXInstanceNormalizationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXInstanceNormalizationOp normOp,
      PatternRewriter &rewriter) const override {
    Location loc = normOp.getLoc();
    Value input = normOp.getInput();
    Value scale = normOp.getScale();
    Value B = normOp.getB();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() < 3)
      return failure();

    int64_t rank = inputType.getRank();

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Create XFEInstanceNormalization operation
    auto normChannelLastOp = rewriter.create<XFEInstanceNormalizationOp>(loc,
        UnrankedTensorType::get(inputType.getElementType()), inputChannelLast,
        scale, B, normOp.getEpsilonAttr());

    // Transpose output back to NCHW
    Value outputNCHW = createOutputTranspose(
        rewriter, loc, normChannelLastOp.getResult(), normOp.getType(), rank);

    rewriter.replaceOp(normOp, outputNCHW);
    return success();
  }
};

// Pattern to convert DepthToSpace to XFEDepthToSpace
struct DepthToSpaceToChannelLastPattern
    : public OpRewritePattern<ONNXDepthToSpaceOp> {
  using OpRewritePattern<ONNXDepthToSpaceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXDepthToSpaceOp d2sOp, PatternRewriter &rewriter) const override {
    Location loc = d2sOp.getLoc();
    Value input = d2sOp.getInput();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    // DepthToSpace requires exactly 4D tensors
    if (!inputType || inputType.getRank() != 4)
      return failure();

    int64_t rank = inputType.getRank();

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Get the original output element type (may be quantized)
    auto origOutputType = mlir::dyn_cast<ShapedType>(d2sOp.getType());
    Type outputElemType = origOutputType ? origOutputType.getElementType()
                                         : inputType.getElementType();

    // Create XFEDepthToSpace operation
    auto d2sChannelLastOp = rewriter.create<XFEDepthToSpaceOp>(loc,
        UnrankedTensorType::get(outputElemType), inputChannelLast,
        d2sOp.getBlocksizeAttr(), d2sOp.getModeAttr());

    // Infer shapes to get ranked type
    if (failed(d2sChannelLastOp.inferShapes([](Region &) {}))) {
      return rewriter.notifyMatchFailure(
          d2sOp, "failed to infer shapes for XFEDepthToSpace");
    }

    // Transpose output back to NCHW
    Value outputNCHW = createOutputTranspose(
        rewriter, loc, d2sChannelLastOp.getResult(), d2sOp.getType(), rank);

    rewriter.replaceOp(d2sOp, outputNCHW);
    return success();
  }
};

// Pattern to convert SpaceToDepth to XFESpaceToDepth
struct SpaceToDepthToChannelLastPattern
    : public OpRewritePattern<ONNXSpaceToDepthOp> {
  using OpRewritePattern<ONNXSpaceToDepthOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSpaceToDepthOp s2dOp, PatternRewriter &rewriter) const override {
    Location loc = s2dOp.getLoc();
    Value input = s2dOp.getInput();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    // SpaceToDepth requires exactly 4D tensors
    if (!inputType || inputType.getRank() != 4)
      return failure();

    int64_t rank = inputType.getRank();

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Get the original output element type (may be quantized)
    auto origOutputType = mlir::dyn_cast<ShapedType>(s2dOp.getType());
    Type outputElemType = origOutputType ? origOutputType.getElementType()
                                         : inputType.getElementType();

    // Create XFESpaceToDepth operation
    auto s2dChannelLastOp = rewriter.create<XFESpaceToDepthOp>(loc,
        UnrankedTensorType::get(outputElemType), inputChannelLast,
        s2dOp.getBlocksizeAttr());

    // Infer shapes to get ranked type
    if (failed(s2dChannelLastOp.inferShapes([](Region &) {}))) {
      return rewriter.notifyMatchFailure(
          s2dOp, "failed to infer shapes for XFESpaceToDepth");
    }

    // Transpose output back to NCHW
    Value outputNCHW = createOutputTranspose(
        rewriter, loc, s2dChannelLastOp.getResult(), s2dOp.getType(), rank);

    rewriter.replaceOp(s2dOp, outputNCHW);
    return success();
  }
};

struct ConvertToChannelLastPass : public PassWrapper<ConvertToChannelLastPass,
                                      OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToChannelLastPass)

  StringRef getArgument() const override { return "convert-to-channel-last"; }

  StringRef getDescription() const override {
    return "Convert ONNX operations to ChannelLast variants with transposes";
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ConvToChannelLastPattern>(context);
    patterns.add<ConvTransposeToChannelLastPattern>(context);
    patterns.add<AveragePoolToChannelLastPattern>(context);
    patterns.add<MaxPoolToChannelLastPattern>(context);
    patterns.add<GlobalAveragePoolToChannelLastPattern>(context);
    patterns.add<GlobalMaxPoolToChannelLastPattern>(context);
    patterns.add<InstanceNormToChannelLastPattern>(context);
    patterns.add<DepthToSpaceToChannelLastPattern>(context);
    patterns.add<SpaceToDepthToChannelLastPattern>(context);

    if (failed(applyPatternsGreedily(function, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createConvertToChannelLastPass() {
  return std::make_unique<ConvertToChannelLastPass>();
}

} // namespace onnx_mlir
