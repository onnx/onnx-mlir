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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

// Helper function to transfer onnx_node_name attribute from source to target op
void transferOnnxNodeName(Operation *sourceOp, Operation *targetOp) {
  if (!sourceOp || !targetOp)
    return;

  // Get onnx_node_name from source operation
  auto onnxNodeName =
      sourceOp->getAttrOfType<mlir::StringAttr>("onnx_node_name");

  // If source has onnx_node_name, set it on target
  if (onnxNodeName && !onnxNodeName.getValue().empty()) {
    targetOp->setAttr("onnx_node_name", onnxNodeName);
  }
}

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

    // Transfer onnx_node_name attribute from original Conv to XFEConv
    transferOnnxNodeName(convOp, convChannelLastOp);

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
      transposeOutputType =
          RankedTensorType::get(origOutputType.getShape(), actualElementType);
    } else {
      transposeOutputType = convOp.getType();
    }

    Value outputNCHW = createOutputTranspose(rewriter, loc,
        convChannelLastOp.getResult(), transposeOutputType, rank);

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

    // Transfer onnx_node_name attribute from original ConvTranspose to
    // XFEConvTranspose
    transferOnnxNodeName(convTransposeOp, convTransposeChannelLastOp);

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
      transposeOutputType =
          RankedTensorType::get(origOutputType.getShape(), actualElementType);
    } else {
      transposeOutputType = convTransposeOp.getType();
    }

    Value outputNCHW = createOutputTranspose(rewriter, loc,
        convTransposeChannelLastOp.getResult(), transposeOutputType, rank);

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
        poolOp.getCountIncludePadAttr(), poolOp.getDilationsAttr(),
        poolOp.getKernelShapeAttr(), poolOp.getPadsAttr(),
        poolOp.getStridesAttr());

    // Transfer onnx_node_name attribute from original AveragePool to
    // XFEAveragePool
    transferOnnxNodeName(poolOp, poolChannelLastOp);

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
    Value outputNCHW = createOutputTranspose(rewriter, loc,
        poolChannelLastOp.getResult(), transposeOutputType, rank);

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

    // Transfer onnx_node_name attribute from original MaxPool to XFEMaxPool
    transferOnnxNodeName(poolOp, poolChannelLastOp);

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
    Value outputNCHW = createOutputTranspose(rewriter, loc,
        poolChannelLastOp.getResult(), transposeOutputType, rank);

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
    auto origOutputType = mlir::cast<ShapedType>(poolOp.getType());
    Type outputElementType = origOutputType.getElementType();
    auto poolChannelLastOp = rewriter.create<XFEGlobalAveragePoolOp>(
        loc, UnrankedTensorType::get(outputElementType), inputChannelLast);

    // Transfer onnx_node_name attribute from original GlobalAveragePool to
    // XFEGlobalAveragePool
    transferOnnxNodeName(poolOp, poolChannelLastOp);

    // CRITICAL: Immediately run shape inference
    if (failed(poolChannelLastOp.inferShapes(nullptr))) {
      return failure();
    }

    // Transpose output back to NCHW
    // Get actual element type from XFEGlobalAveragePool output
    auto xfeOutputType =
        mlir::cast<ShapedType>(poolChannelLastOp.getResult().getType());
    auto transposeOutputType = RankedTensorType::get(
        origOutputType.getShape(), xfeOutputType.getElementType());
    Value outputNCHW = createOutputTranspose(rewriter, loc,
        poolChannelLastOp.getResult(), transposeOutputType, rank);

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
    auto origOutputType = mlir::cast<ShapedType>(poolOp.getType());
    Type outputElementType = origOutputType.getElementType();
    auto poolChannelLastOp = rewriter.create<XFEGlobalMaxPoolOp>(
        loc, UnrankedTensorType::get(outputElementType), inputChannelLast);

    // Transfer onnx_node_name attribute from original GlobalMaxPool to
    // XFEGlobalMaxPool
    transferOnnxNodeName(poolOp, poolChannelLastOp);

    // CRITICAL: Immediately run shape inference
    if (failed(poolChannelLastOp.inferShapes(nullptr))) {
      return failure();
    }

    // Transpose output back to NCHW
    // Get actual element type from XFEGlobalMaxPool output
    auto xfeOutputType =
        mlir::cast<ShapedType>(poolChannelLastOp.getResult().getType());
    auto transposeOutputType = RankedTensorType::get(
        origOutputType.getShape(), xfeOutputType.getElementType());
    Value outputNCHW = createOutputTranspose(rewriter, loc,
        poolChannelLastOp.getResult(), transposeOutputType, rank);

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
    // Use original norm's output element type to preserve quantization
    auto origOutputType = mlir::cast<ShapedType>(normOp.getType());
    Type outputElementType = origOutputType.getElementType();
    auto normChannelLastOp = rewriter.create<XFEInstanceNormalizationOp>(loc,
        UnrankedTensorType::get(outputElementType), inputChannelLast, scale, B,
        normOp.getEpsilonAttr());

    // Transfer onnx_node_name attribute from original InstanceNormalization to
    // XFEInstanceNormalization
    transferOnnxNodeName(normOp, normChannelLastOp);

    // CRITICAL: Immediately run shape inference
    if (failed(normChannelLastOp.inferShapes(nullptr))) {
      return failure();
    }

    // Transpose output back to NCHW
    // Get actual element type from XFEInstanceNormalization output
    auto xfeOutputType =
        mlir::cast<ShapedType>(normChannelLastOp.getResult().getType());
    auto transposeOutputType = RankedTensorType::get(
        origOutputType.getShape(), xfeOutputType.getElementType());
    Value outputNCHW = createOutputTranspose(rewriter, loc,
        normChannelLastOp.getResult(), transposeOutputType, rank);

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

    // Transfer onnx_node_name attribute from original DepthToSpace to
    // XFEDepthToSpace
    transferOnnxNodeName(d2sOp, d2sChannelLastOp);

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

    // Transfer onnx_node_name attribute from original SpaceToDepth to
    // XFESpaceToDepth
    transferOnnxNodeName(s2dOp, s2dChannelLastOp);

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

// Pattern to convert Resize to XFEResize
// Only converts when resize is on spatial dimensions (NCHW layout assumed):
// - Batch dimension (dim 0) must not be resized
// - Channel dimension (dim 1) must not be resized
// - Only spatial dimensions (dims 2, 3) should be resized
//
// TODO: Currently only handles 4D tensors (NCHW/NHWC format).
//       Future work needed to support:
//       - 3D tensors (NCW/NWC for 1D spatial)
//       - 5D tensors (NCDHW/NDHWC for 3D spatial)
//       - General N-dimensional cases
struct ResizeToChannelLastPattern : public OpRewritePattern<ONNXResizeOp> {
  using OpRewritePattern<ONNXResizeOp>::OpRewritePattern;

  // Check if resize is NCHW spatial-only by comparing input/output shapes
  // For NCHW layout: dim 0 = batch, dim 1 = channel, dims 2,3 = spatial (H,W)
  // A spatial-only resize keeps dims 0 and 1 unchanged
  // TODO: Extend to support 3D (NCW) and 5D (NCDHW) tensors
  static bool isNCHWSpatialResize(
      RankedTensorType inputType, RankedTensorType outputType) {
    if (!inputType || !outputType)
      return false;

    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    // Must have same rank
    if (inputShape.size() != outputShape.size())
      return false;

    int64_t rank = inputShape.size();
    if (rank != 4)
      return false; // Only support 4D tensors (NCHW)

    // Check dim 0 (batch) - must be unchanged
    // Allow dynamic dims (they match by definition for this check)
    if (inputShape[0] != ShapedType::kDynamic &&
        outputShape[0] != ShapedType::kDynamic &&
        inputShape[0] != outputShape[0]) {
      return false;
    }

    // Check dim 1 (channel in NCHW) - must be unchanged
    if (inputShape[1] != ShapedType::kDynamic &&
        outputShape[1] != ShapedType::kDynamic &&
        inputShape[1] != outputShape[1]) {
      return false;
    }

    // Dims 0 and 1 are unchanged - this is NCHW spatial resize
    return true;
  }

  // Check if resize appears to be NHWC spatial by comparing input/output shapes
  // For NHWC layout: dim 0 = batch, dims 1,2 = spatial (H,W), dim 3 = channel
  // A spatial-only NHWC resize keeps dims 0 and 3 unchanged, dims 1,2 change
  // TODO: Extend to support 3D (NWC) and 5D (NDHWC) tensors
  static bool isNHWCSpatialResize(
      RankedTensorType inputType, RankedTensorType outputType) {
    if (!inputType || !outputType)
      return false;

    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    if (inputShape.size() != outputShape.size())
      return false;

    int64_t rank = inputShape.size();
    if (rank != 4)
      return false; // Only support 4D tensors (NHWC)

    // For NHWC [N, H, W, C]: dim 0 = batch, dims 1,2 = spatial, dim 3 = channel
    // Check dim 0 (batch) - must be unchanged
    if (inputShape[0] != ShapedType::kDynamic &&
        outputShape[0] != ShapedType::kDynamic &&
        inputShape[0] != outputShape[0]) {
      return false;
    }

    // Check dim 3 (channel in NHWC) - must be unchanged
    if (inputShape[3] != ShapedType::kDynamic &&
        outputShape[3] != ShapedType::kDynamic &&
        inputShape[3] != outputShape[3]) {
      return false;
    }

    // Check if spatial dims (1 or 2) actually change
    bool spatialChanges = false;
    if ((inputShape[1] != ShapedType::kDynamic &&
            outputShape[1] != ShapedType::kDynamic &&
            inputShape[1] != outputShape[1]) ||
        (inputShape[2] != ShapedType::kDynamic &&
            outputShape[2] != ShapedType::kDynamic &&
            inputShape[2] != outputShape[2])) {
      spatialChanges = true;
    }

    // It's NHWC spatial resize if batch and channel unchanged, spatial changes
    return spatialChanges;
  }

  LogicalResult matchAndRewrite(
      ONNXResizeOp resizeOp, PatternRewriter &rewriter) const override {
    Location loc = resizeOp.getLoc();
    Value input = resizeOp.getX();

    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    // TODO: Currently only supports 4D tensors (NCHW/NHWC format).
    //       Extend to handle 3D (NCW/NWC) and 5D (NCDHW/NDHWC) in future.
    if (!inputType || inputType.getRank() != 4)
      return failure();

    int64_t rank = inputType.getRank();

    // Get output type to compare shapes
    auto outputType = mlir::dyn_cast<RankedTensorType>(resizeOp.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(
          resizeOp, "Output type is not ranked. Cannot determine layout.");
    }

    // CHECK 1: Is this an NHWC spatial resize?
    // If dims 0 (batch) and last (channel) are unchanged but middle dims
    // change, the input is already in NHWC layout - don't convert
    if (isNHWCSpatialResize(inputType, outputType)) {
      return rewriter.notifyMatchFailure(resizeOp,
          "Resize appears to be NHWC spatial (batch and last dim unchanged, "
          "middle dims changed). Input is likely already channel-last.");
    }

    // CHECK 2: Is this an NCHW spatial resize?
    // If dims 0 (batch) and 1 (channel) are unchanged, it's NCHW spatial -
    // convert
    if (!isNCHWSpatialResize(inputType, outputType)) {
      return rewriter.notifyMatchFailure(resizeOp,
          "Resize is not NCHW spatial-only (batch or channel dimensions "
          "are being resized). Cannot convert to channel-last layout.");
    }

    // Transpose input to channel-last
    Value inputChannelLast = createInputTranspose(
        rewriter, loc, input, rank, inputType.getElementType());

    // Get the original output element type (may be quantized)
    auto origOutputType = mlir::dyn_cast<ShapedType>(resizeOp.getType());
    Type outputElemType = origOutputType ? origOutputType.getElementType()
                                         : inputType.getElementType();

    // Handle scales and sizes - need to permute them for channel-last layout
    // Permutation: [0, 2, 3, ..., N-1, 1] for NCHW -> NHWC
    Value roi = resizeOp.getRoi();
    Value scales = resizeOp.getScales();
    Value sizes = resizeOp.getSizes();

    // Helper to permute a 1D tensor from NCHW order to NHWC order
    // CRITICAL: Must create a constant-folded result for shape inference to
    // work
    auto permuteForChannelLast = [&](Value tensor) -> Value {
      if (isa<NoneType>(tensor.getType()))
        return tensor;

      auto tensorType = mlir::dyn_cast<RankedTensorType>(tensor.getType());
      if (!tensorType || !tensorType.hasStaticShape())
        return tensor;

      int64_t numElements = tensorType.getNumElements();
      if (numElements != rank)
        return tensor; // Can't permute if size doesn't match rank

      // Try to get constant values for folding
      DenseElementsAttr constAttr;
      if (auto defOp = tensor.getDefiningOp<ONNXConstantOp>()) {
        if (auto valueAttr = defOp.getValue()) {
          constAttr = mlir::dyn_cast<DenseElementsAttr>(*valueAttr);
        }
      }

      // Create permutation for channel-last: [0, 2, 3, ..., N-1, 1]
      SmallVector<int64_t, 4> perm;
      perm.push_back(0); // batch
      for (int64_t i = 2; i < rank; ++i)
        perm.push_back(i); // spatial dimensions
      perm.push_back(1);   // channels

      // If input is a constant, create a permuted constant (constant folding)
      if (constAttr) {
        auto elemType = tensorType.getElementType();
        if (elemType.isF32()) {
          // Permute float constants
          auto values = constAttr.getValues<float>();
          SmallVector<float, 4> permutedValues;
          for (int64_t p : perm) {
            permutedValues.push_back(values[p]);
          }
          auto permutedAttr = DenseElementsAttr::get(
              tensorType, llvm::ArrayRef<float>(permutedValues));
          return rewriter.create<ONNXConstantOp>(
              loc, mlir::Attribute(), permutedAttr);
        } else if (elemType.isInteger(64)) {
          // Permute int64 constants
          auto values = constAttr.getValues<int64_t>();
          SmallVector<int64_t, 4> permutedValues;
          for (int64_t p : perm) {
            permutedValues.push_back(values[p]);
          }
          auto permutedAttr =
              DenseIntElementsAttr::get(tensorType, permutedValues);
          return rewriter.create<ONNXConstantOp>(
              loc, mlir::Attribute(), permutedAttr);
        }
      }

      // Fallback: Create Gather op for non-constant inputs
      SmallVector<int64_t, 4> indices(perm.begin(), perm.end());
      auto indicesType = RankedTensorType::get({rank}, rewriter.getI64Type());
      auto indicesAttr = DenseIntElementsAttr::get(indicesType, indices);
      Value indicesValue =
          rewriter.create<ONNXConstantOp>(loc, mlir::Attribute(), indicesAttr);

      auto si64Type = rewriter.getIntegerType(64, /*isSigned=*/true);
      return rewriter.create<ONNXGatherOp>(loc, tensorType, tensor,
          indicesValue, rewriter.getIntegerAttr(si64Type, 0));
    };

    Value roiChannelLast = permuteForChannelLast(roi);
    Value scalesChannelLast = permuteForChannelLast(scales);
    Value sizesChannelLast = permuteForChannelLast(sizes);

    // Create XFEResize operation
    auto resizeChannelLastOp = rewriter.create<XFEResizeOp>(loc,
        UnrankedTensorType::get(outputElemType), inputChannelLast,
        roiChannelLast, scalesChannelLast, sizesChannelLast,
        resizeOp.getAntialiasAttr(), resizeOp.getAxesAttr(),
        resizeOp.getCoordinateTransformationModeAttr(),
        resizeOp.getCubicCoeffAAttr(), resizeOp.getExcludeOutsideAttr(),
        resizeOp.getExtrapolationValueAttr(),
        resizeOp.getKeepAspectRatioPolicyAttr(), resizeOp.getModeAttr(),
        resizeOp.getNearestModeAttr());

    // Transfer onnx_node_name attribute from original Resize to XFEResize
    transferOnnxNodeName(resizeOp, resizeChannelLastOp);

    // Infer shapes to get ranked type
    if (failed(resizeChannelLastOp.inferShapes([](Region &) {}))) {
      return rewriter.notifyMatchFailure(
          resizeOp, "failed to infer shapes for XFEResize");
    }

    // Transpose output back to NCHW
    Value outputNCHW = createOutputTranspose(rewriter, loc,
        resizeChannelLastOp.getResult(), resizeOp.getType(), rank);

    rewriter.replaceOp(resizeOp, outputNCHW);
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

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ONNXDialect>();
    registry.insert<func::FuncDialect>();
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
    patterns.add<ResizeToChannelLastPattern>(context);

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
