// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

#include "llvm/ADT/SmallVector.h"

#include <cstring>
#include <optional>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get the static shape from a tensor type, returns nullopt if dynamic.
std::optional<SmallVector<int64_t>> getStaticShape(Type type) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape())
    return std::nullopt;
  return SmallVector<int64_t>(tensorType.getShape());
}

/// Create an expanded constant by repeating the original constant data.
/// Supports float, integer, and quantized element types via raw byte access.
ONNXConstantOp createExpandedConstant(PatternRewriter &rewriter, Location loc,
    Value originalConst, int64_t targetSize, int64_t /*originalSize*/) {
  // Try to get the constant op or from scast
  auto constOp = originalConst.getDefiningOp<ONNXConstantOp>();
  if (!constOp) {
    // Try to get from scast
    if (auto scastOp = originalConst.getDefiningOp<quant::StorageCastOp>()) {
      constOp = scastOp.getOperand().getDefiningOp<ONNXConstantOp>();
    }
    if (!constOp)
      return nullptr;
  }

  auto attr = mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  if (!attr)
    return nullptr;

  // Get the original element type from the Value (preserves quantized type)
  auto origType = mlir::cast<ShapedType>(originalConst.getType());
  Type origElemType = origType.getElementType();

  // Get storage type for DenseElementsAttr (i8 for quantized, same for others)
  Type storageType = attr.getElementType();

  auto storageTensorType = RankedTensorType::get({targetSize}, storageType);

  // Result type uses the original element type (preserves quantized info)
  auto resultType = RankedTensorType::get({targetSize}, origElemType);

  DenseElementsAttr newAttr;
  if (attr.isSplat()) {
    // For splat attributes, repeating the same value produces another splat.
    newAttr = DenseElementsAttr::get(
        storageTensorType, attr.getSplatValue<Attribute>());
  } else {
    // For non-splat attributes, iterate and copy each element
    unsigned bitWidth = storageType.getIntOrFloatBitWidth();
    unsigned byteWidth = (bitWidth + 7) / 8;

    // Get raw data from original constant
    auto rawData = attr.getRawData();
    int64_t origNumElements = attr.getNumElements();

    // Expand data by repeating (same logic as expandConstantData)
    std::vector<char> expandedData(targetSize * byteWidth, 0);

    for (int64_t i = 0; i < targetSize; ++i) {
      int64_t srcIdx = i % origNumElements;
      std::memcpy(&expandedData[i * byteWidth],
          &rawData.data()[srcIdx * byteWidth], byteWidth);
    }

    // Create DenseElementsAttr with storage type
    newAttr = DenseElementsAttr::getFromRawBuffer(
        storageTensorType, ArrayRef<char>(expandedData));
  }

  // Create new constant op with expanded data
  return rewriter.create<ONNXConstantOp>(loc, resultType,
      /*sparse_value=*/Attribute(),
      /*value=*/newAttr,
      /*value_float=*/FloatAttr(),
      /*value_floats=*/ArrayAttr(),
      /*value_int=*/IntegerAttr(),
      /*value_ints=*/ArrayAttr(),
      /*value_string=*/StringAttr(),
      /*value_strings=*/ArrayAttr());
}

//===----------------------------------------------------------------------===//
// Pattern: MergeReshapeInstanceNormPattern
//===----------------------------------------------------------------------===//

/// Pattern to match:
///   input -> Reshape -> InstanceNormalization -> Reshape
///
/// And convert to:
///   input -> GroupNormalization
///
struct MergeReshapeInstanceNormPattern
    : public OpRewritePattern<ONNXReshapeOp> {

  using OpRewritePattern<ONNXReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReshapeOp bottomReshape, PatternRewriter &rewriter) const override {

    // Match the pattern bottom-up from the final reshape
    // Pattern: Reshape <- InstanceNorm <- Reshape

    // 1. Get InstanceNormalization before bottom reshape
    auto instanceNorm =
        bottomReshape.getData().getDefiningOp<ONNXInstanceNormalizationOp>();
    if (!instanceNorm)
      return failure();

    // 2. Get top Reshape
    auto topReshape = instanceNorm.getInput().getDefiningOp<ONNXReshapeOp>();
    if (!topReshape)
      return failure();

    // 3. Get the original input (before top reshape)
    Value originalInput = topReshape.getData();

    // Validate shapes
    auto inputShapeOpt = getStaticShape(originalInput.getType());
    auto outputShapeOpt = getStaticShape(bottomReshape.getResult().getType());
    auto instanceShapeOpt = getStaticShape(instanceNorm.getResult().getType());

    if (!inputShapeOpt || !outputShapeOpt || !instanceShapeOpt)
      return rewriter.notifyMatchFailure(
          bottomReshape, "Cannot get static shapes");

    auto inputShape = *inputShapeOpt;
    auto outputShape = *outputShapeOpt;
    auto instanceShape = *instanceShapeOpt;

    // Input and output shapes must match (reshapes cancel out)
    if (inputShape != outputShape)
      return rewriter.notifyMatchFailure(
          bottomReshape, "Input/output shapes don't match");

    // Only support 3D (NCD) or 4D (NCHW) tensors
    if (inputShape.size() != 3 && inputShape.size() != 4)
      return rewriter.notifyMatchFailure(
          bottomReshape, "Only 3D/4D tensors supported, got " +
                             std::to_string(inputShape.size()) + "D");

    // Calculate group count.
    // GroupNorm is decomposed as Reshape → InstanceNorm → Reshape using the
    // spatial-merge approach: [N, C, H, W] → [N, G, (C/G)*H, W].
    // InstanceNorm's channel dim (axis 1) IS the number of groups G, because
    // the C/G channels are merged into the spatial dimensions.
    int64_t gnChannel = inputShape[1];    // Original channel count (C)
    int64_t inChannel = instanceShape[1]; // InstanceNorm channel count (G)

    if (inChannel == 0 || gnChannel < inChannel || gnChannel % inChannel != 0)
      return rewriter.notifyMatchFailure(
          bottomReshape, "Invalid channel relationship: gnChannel=" +
                             std::to_string(gnChannel) +
                             ", inChannel=" + std::to_string(inChannel));

    int64_t numGroups = inChannel;

    // Get scale and bias from InstanceNorm
    Value instanceScale = instanceNorm.getScale();
    Value instanceBias = instanceNorm.getB();
    float epsilon = instanceNorm.getEpsilon().convertToFloat();

    // Create expanded scale constant
    auto newScale = createExpandedConstant(
        rewriter, bottomReshape.getLoc(), instanceScale, gnChannel, inChannel);
    if (!newScale)
      return rewriter.notifyMatchFailure(
          bottomReshape, "Failed to create new scale");

    // Create expanded bias constant
    auto newBias = createExpandedConstant(
        rewriter, bottomReshape.getLoc(), instanceBias, gnChannel, inChannel);
    if (!newBias) {
      // Rollback changes when failing, otherwise the pattern matcher will be
      // stuck in infinite loop
      rewriter.eraseOp(newScale);
      return rewriter.notifyMatchFailure(
          bottomReshape, "Failed to create new bias");
    }

    // Create GroupNormalization op

    auto resultType = bottomReshape.getResult().getType();

    // Build attributes for GroupNorm
    auto epsilonAttr = rewriter.getF32FloatAttr(epsilon);
    auto numGroupsAttr = IntegerAttr::get(
        rewriter.getIntegerType(64, /*isSigned=*/true), numGroups);

    // Create GroupNorm op
    auto groupNormOp = rewriter.create<ONNXGroupNormalizationOp>(
        bottomReshape.getLoc(), resultType, originalInput, newScale, newBias,
        epsilonAttr, numGroupsAttr);

    // Replace all uses of the bottom reshape with the GroupNorm result
    rewriter.replaceOp(bottomReshape, groupNormOp.getResult());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct ConvertInstanceNormToGroupNormPass
    : public PassWrapper<ConvertInstanceNormToGroupNormPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "convert-instancenorm-to-groupnorm";
  }
  StringRef getDescription() const override {
    return "Convert InstanceNormalization to GroupNormalization";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<MergeReshapeInstanceNormPattern>(context);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertInstanceNormToGroupNormPass() {
  return std::make_unique<ConvertInstanceNormToGroupNormPass>();
}

} // namespace onnx_mlir
