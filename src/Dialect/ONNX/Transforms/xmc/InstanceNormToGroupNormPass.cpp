// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cstring>
#include <optional>

using namespace mlir;

#define DEBUG_TYPE "convert-instancenorm-to-groupnorm"

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

/// Check if two shapes are equal.
bool shapesEqual(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2) {
  if (shape1.size() != shape2.size())
    return false;
  for (size_t i = 0; i < shape1.size(); ++i) {
    if (shape1[i] != shape2[i])
      return false;
  }
  return true;
}

/// Expand constant data by repeating it to fill the target size.
/// Used to expand InstanceNorm scale/bias to GroupNorm scale/bias.
template <typename T>
SmallVector<T> expandConstantData(
    ArrayRef<T> original, int64_t targetSize) {
  SmallVector<T> expanded;
  expanded.reserve(targetSize);
  while (static_cast<int64_t>(expanded.size()) < targetSize) {
    expanded.append(original.begin(), original.end());
  }
  return expanded;
}

/// Extract constant data from an ONNX constant op.
template <typename T>
std::optional<SmallVector<T>> getConstantData(Value value) {
  auto constOp = value.getDefiningOp<ONNXConstantOp>();
  if (!constOp)
    return std::nullopt;

  auto attr = mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  if (!attr)
    return std::nullopt;

  SmallVector<T> data;
  for (auto val : attr.getValues<T>()) {
    data.push_back(val);
  }
  return data;
}

/// Create an expanded constant by repeating the original constant data.
/// Supports float, integer, and quantized element types via raw byte access.
Value createExpandedConstant(PatternRewriter &rewriter, Location loc,
    Value originalConst, int64_t targetSize, int64_t /*originalSize*/) {
  // Try to get the constant op
  auto constOp = originalConst.getDefiningOp<ONNXConstantOp>();

  if (!constOp)
    return nullptr;

  auto attr = mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  if (!attr)
    return nullptr;

  // Get the original element type from the Value (preserves quantized type)
  auto origType = mlir::cast<ShapedType>(originalConst.getType());
  Type origElemType = origType.getElementType();

  // Get storage type for DenseElementsAttr (i8 for quantized, same for others)
  Type storageType = attr.getElementType();

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
  auto storageTensorType = RankedTensorType::get({targetSize}, storageType);
  auto newAttr = DenseElementsAttr::getFromRawBuffer(
      storageTensorType, ArrayRef<char>(expandedData));

  // Result type uses the original element type (preserves quantized info)
  auto resultType = RankedTensorType::get({targetSize}, origElemType);

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
    LLVM_DEBUG(llvm::dbgs() << "Checking reshape op for IN->GN pattern\n");

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

    if (!inputShapeOpt || !outputShapeOpt || !instanceShapeOpt) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot get static shapes\n");
      return failure();
    }

    auto inputShape = *inputShapeOpt;
    auto outputShape = *outputShapeOpt;
    auto instanceShape = *instanceShapeOpt;

    // Input and output shapes must match (reshapes cancel out)
    if (!shapesEqual(inputShape, outputShape)) {
      LLVM_DEBUG(llvm::dbgs() << "Input/output shapes don't match, skipping\n");
      return failure();
    }

    // Only support 3D (NCD) or 4D (NCHW) tensors
    if (inputShape.size() != 3 && inputShape.size() != 4) {
      LLVM_DEBUG(llvm::dbgs() << "Only 3D/4D tensors supported, got "
                              << inputShape.size() << "D\n");
      return failure();
    }

    // Calculate group count
    int64_t gnChannel = inputShape[1];    // Original channel count
    int64_t inChannel = instanceShape[1]; // InstanceNorm channel count

    if (inChannel == 0 || gnChannel < inChannel || gnChannel % inChannel != 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Invalid channel relationship: gnChannel=" << gnChannel
                 << ", inChannel=" << inChannel << "\n");
      return failure();
    }

    int64_t numGroups = gnChannel / inChannel;

    LLVM_DEBUG(llvm::dbgs() << "Converting InstanceNorm to GroupNorm with "
                            << numGroups << " groups\n");

    // Get scale and bias from InstanceNorm
    Value instanceScale = instanceNorm.getScale();
    Value instanceBias = instanceNorm.getB();
    float epsilon = instanceNorm.getEpsilon().convertToFloat();

    // Create expanded scale constant
    Value newScale = createExpandedConstant(
        rewriter, bottomReshape.getLoc(), instanceScale, gnChannel, inChannel);
    if (!newScale)
      return failure();

    // Create expanded bias constant
    Value newBias = createExpandedConstant(
        rewriter, bottomReshape.getLoc(), instanceBias, gnChannel, inChannel);
    if (!newBias)
      return failure();

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

    LLVM_DEBUG(
        llvm::dbgs() << "Successfully converted InstanceNorm to GroupNorm\n");
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
    config.maxIterations = 3;

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
