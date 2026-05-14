//===- LowerReduceToPoolIPUPass.cpp - Lower Reduce to Pool for IPU --------===//
//
// Copyright (C) 2019 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <optional>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Constants - Pool Engine Limits
//===----------------------------------------------------------------------===//

constexpr int64_t MAX_KERNEL_SIZE = 16;
constexpr int64_t MAX_SQUARE_KERNEL_DIM = 8; // For square kernels (8x8)
//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Extract constant integer values from axes attribute
std::optional<SmallVector<int64_t>> extractAxes(Attribute axesAttr) {
  if (!axesAttr)
    return std::nullopt;

  if (auto arrayAttr = dyn_cast<ArrayAttr>(axesAttr)) {
    SmallVector<int64_t> axes;
    for (auto elem : arrayAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(elem)) {
        axes.push_back(intAttr.getInt());
      } else {
        return std::nullopt;
      }
    }
    return axes;
  }
  return std::nullopt;
}

/// Extract axes from ReduceMax's axes input (it's a tensor input, not
/// attribute)
std::optional<SmallVector<int64_t>> extractAxesFromInput(Value axesValue) {
  if (!axesValue)
    return std::nullopt;

  auto defOp = axesValue.getDefiningOp<ONNXConstantOp>();
  if (!defOp)
    return std::nullopt;

  auto valueAttr = defOp.getValueAttr();
  if (!valueAttr)
    return std::nullopt;

  auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr);
  if (!denseAttr)
    return std::nullopt;

  SmallVector<int64_t> axes;
  for (auto val : denseAttr.getValues<APInt>()) {
    axes.push_back(val.getSExtValue());
  }
  return axes;
}

/// Normalize negative axes to positive
void normalizeAxes(SmallVector<int64_t> &axes, int64_t rank) {
  for (auto &axis : axes) {
    if (axis < 0) {
      axis += rank;
    }
  }
  std::sort(axes.begin(), axes.end());
}

/// Remove trivial axes (dimension size = 1)
SmallVector<int64_t> removeTrivialAxes(
    ArrayRef<int64_t> axes, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> result;
  for (auto axis : axes) {
    if (axis >= 0 && axis < static_cast<int64_t>(shape.size()) &&
        shape[axis] != 1) {
      result.push_back(axis);
    }
  }
  return result;
}

/// Check if axes are continuous
bool areAxesContinuous(ArrayRef<int64_t> axes) {
  if (axes.size() <= 1)
    return true;
  for (size_t i = 1; i < axes.size(); ++i) {
    if (axes[i] != axes[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

/// Check if axes are valid for SPATIAL pooling only (not batch or channel)
/// Layout: NCHW - batch=0, channel=1, spatial=2,3,...
bool areAxesValidForSpatialPooling(ArrayRef<int64_t> axes, int64_t /*rank*/) {
  if (axes.empty())
    return false;

  for (auto axis : axes) {
    if (axis == 0 || axis == 1) {
      return false; // Batch (0) or channel (1) dimension
    }
  }
  return areAxesContinuous(axes);
}

/// Check if reduction includes channel dimension (index 1 in NCHW)
bool includesChannelDimension(ArrayRef<int64_t> axes, int64_t /*rank*/) {
  return std::find(axes.begin(), axes.end(), 1) != axes.end();
}

/// Calculate pool parameters for SPATIAL reduction
/// Layout: NCHW - [batch, channels, height, width]
std::tuple<SmallVector<int64_t>, int64_t, int64_t>
calculateSpatialPoolParameters(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> reductionAxes) {
  int64_t rank = inputShape.size();
  // NCHW: C is at index 1
  int64_t N = inputShape[0];
  int64_t C = (rank > 1) ? inputShape[1] : 1;
  int64_t H = 1;
  int64_t W = 1;
  int64_t kernelH = 1;
  int64_t kernelW = 1;

  int64_t lastAxis = reductionAxes.back();

  // Process spatial dimensions (indices 2 and beyond)
  for (int64_t idx = 2; idx < rank; ++idx) {
    bool isReductionAxis = std::find(reductionAxes.begin(), reductionAxes.end(),
                               idx) != reductionAxes.end();

    if (isReductionAxis) {
      if (kernelH * inputShape[idx] <= MAX_KERNEL_SIZE) {
        H *= inputShape[idx];
        kernelH *= inputShape[idx];
      } else {
        W *= inputShape[idx];
        kernelW *= inputShape[idx];
      }
    } else if (idx > lastAxis) {
      W *= inputShape[idx];
    }
  }

  // Try square kernel
  int64_t totalKernel = kernelH * kernelW;
  auto sqrtInt =
      static_cast<int64_t>(std::sqrt(static_cast<double>(totalKernel)));
  if (sqrtInt * sqrtInt == totalKernel && sqrtInt <= MAX_SQUARE_KERNEL_DIM) {
    kernelH = kernelW = sqrtInt;
    H = kernelH;
    W = kernelW;
  }

  // Return shape in NCHW format
  return {{N, C, H, W}, kernelH, kernelW};
}

/// Calculate pool parameters for CHANNEL reduction (reshape trick)
/// Layout: NCHW - channels are moved to spatial dimension for pooling
std::tuple<SmallVector<int64_t>, int64_t, int64_t>
calculateChannelPoolParameters(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> reductionAxes) {
  int64_t rank = inputShape.size();

  // For channel reduction in NCHW: [N, C, H, W] → [1, 1, H*W*N, C] then pool
  int64_t C = 1; // Dimensions to reduce become the kernel width
  int64_t W = 1; // All other dimensions go here

  for (int64_t idx = 0; idx < rank; ++idx) {
    bool isReductionAxis = std::find(reductionAxes.begin(), reductionAxes.end(),
                               idx) != reductionAxes.end();

    if (isReductionAxis) {
      C *= inputShape[idx]; // Dimensions to reduce → becomes kernel
    } else {
      W *= inputShape[idx]; // Other dimensions → width
    }
  }

  // Pool shape in NCHW: [1, 1, W, C] - pool across last dimension
  // Kernel: [1, C] - takes max/avg across all reduced values
  int64_t kernelH = 1;
  int64_t kernelW = C;

  return {{1, 1, W, C}, kernelH, kernelW};
}

/// Create a reshape operation
Value createReshapeOp(PatternRewriter &rewriter, Location loc, Value input,
    ArrayRef<int64_t> newShape, Type elementType) {
  auto newType = RankedTensorType::get(newShape, elementType);
  auto shapeSize = static_cast<int64_t>(newShape.size());
  auto shapeType = RankedTensorType::get({shapeSize}, rewriter.getI64Type());
  auto shapeAttr =
      DenseElementsAttr::get(shapeType, llvm::ArrayRef<int64_t>(newShape));

  auto shapeConst = rewriter.create<ONNXConstantOp>(loc, shapeType, Attribute(),
      shapeAttr, FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(),
      StringAttr(), ArrayAttr());

  return rewriter
      .create<ONNXReshapeOp>(loc, newType, input, shapeConst, IntegerAttr())
      .getReshaped();
}

/// Normalize and filter axes, handling trivial reduction case.
/// Returns nullopt if axes become empty after filtering (trivial reduction).
/// When nullopt is returned, the reduce op is replaced with input or reshape.
template <typename ReduceOpType>
std::optional<SmallVector<int64_t>> normalizeAndFilterAxes(
    SmallVector<int64_t> axes, int64_t rank, ArrayRef<int64_t> inputShape,
    ReduceOpType reduceOp, RankedTensorType inputType,
    RankedTensorType outputType, PatternRewriter &rewriter, Location loc) {
  normalizeAxes(axes, rank);
  axes = removeTrivialAxes(axes, inputShape);

  if (axes.empty()) {
    if (inputType.getShape() == outputType.getShape()) {
      rewriter.replaceOp(reduceOp, reduceOp.getData());
    } else {
      Value reshaped = createReshapeOp(rewriter, loc, reduceOp.getData(),
          outputType.getShape(), outputType.getElementType());
      rewriter.replaceOp(reduceOp, reshaped);
    }
    return std::nullopt;
  }

  return axes;
}

//===----------------------------------------------------------------------===//
/// Pattern: ReduceMeanV13 → AveragePool
///
/// ONNXReduceMeanV13Op uses attribute-based axes (I64ArrayAttr).
/// onnx-mlir canonicalises GlobalAveragePool into ReduceMeanV13 with
/// axes=[2,3,...,rank-1], so this pattern is the primary entry point for
/// converting global-average-pool semantics into an ONNXAveragePoolOp
/// that downstream XIR lowering already handles.
//===----------------------------------------------------------------------===//

struct LowerReduceMeanV13ToAvgPoolPattern
    : public OpRewritePattern<ONNXReduceMeanV13Op> {
  using OpRewritePattern<ONNXReduceMeanV13Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceMeanV13Op reduceOp, PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();

    auto inputType = dyn_cast<RankedTensorType>(reduceOp.getData().getType());
    auto outputType = dyn_cast<RankedTensorType>(reduceOp.getType());

    if (!inputType || !outputType || !inputType.hasStaticShape())
      return failure();

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    if (rank > 4 || rank < 3)
      return failure();

    auto axesOpt = extractAxes(reduceOp.getAxesAttr());
    if (!axesOpt)
      return failure();

    auto axesResult = normalizeAndFilterAxes(*axesOpt, rank, inputShape,
        reduceOp, inputType, outputType, rewriter, loc);
    if (!axesResult)
      return success();

    SmallVector<int64_t> axes = *axesResult;

    if (!areAxesValidForSpatialPooling(axes, rank))
      return failure();

    auto [poolShape, kernelH, kernelW] =
        calculateSpatialPoolParameters(inputShape, axes);

    Value poolInput = reduceOp.getData();
    if (inputShape != ArrayRef<int64_t>(poolShape)) {
      poolInput = createReshapeOp(rewriter, loc, reduceOp.getData(), poolShape,
          inputType.getElementType());
    }

    auto kernelShapeAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto stridesAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto padsAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});

    int64_t outH = (poolShape[2] + kernelH - 1) / kernelH;
    int64_t outW = (poolShape[3] + kernelW - 1) / kernelW;
    SmallVector<int64_t> poolOutputShape = {
        poolShape[0], poolShape[1], outH, outW};
    auto poolOutputType =
        RankedTensorType::get(poolOutputShape, inputType.getElementType());

    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);
    auto avgPoolOp =
        rewriter.create<ONNXAveragePoolOp>(loc, poolOutputType, poolInput,
            /*auto_pad=*/rewriter.getStringAttr("NOTSET"),
            /*ceil_mode=*/IntegerAttr::get(si64Type, 1),
            /*count_include_pad=*/IntegerAttr::get(si64Type, 0),
            /*dilations=*/nullptr, kernelShapeAttr, padsAttr, stridesAttr);

    Value result = avgPoolOp.getResult();

    if (poolOutputShape != outputType.getShape()) {
      result = createReshapeOp(rewriter, loc, result, outputType.getShape(),
          outputType.getElementType());
    }

    rewriter.replaceOp(reduceOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
/// Pattern: ReduceMean → AveragePool
//===----------------------------------------------------------------------===//

struct LowerReduceMeanToAvgPoolPattern
    : public OpRewritePattern<ONNXReduceMeanOp> {
  using OpRewritePattern<ONNXReduceMeanOp>::OpRewritePattern;
  /// match and rewrite the ReduceMean op to AveragePool op
  LogicalResult matchAndRewrite(
      ONNXReduceMeanOp reduceOp, PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();

    auto inputType = dyn_cast<RankedTensorType>(reduceOp.getData().getType());
    auto outputType = dyn_cast<RankedTensorType>(reduceOp.getType());

    if (!inputType || !outputType || !inputType.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    if (rank > 4 || rank < 3) {
      return failure();
    }

    // Extract axes
    auto axesOpt = extractAxesFromInput(reduceOp.getAxes());
    if (!axesOpt)
      return failure();

    auto axesResult = normalizeAndFilterAxes(*axesOpt, rank, inputShape,
        reduceOp, inputType, outputType, rewriter, loc);
    if (!axesResult)
      return success(); // Trivial reduction handled

    SmallVector<int64_t> axes = *axesResult;

    // Defer to ReplaceQDQReductionPass for single-axis reductions it can
    // canonicalise to a rank-4 + keep_dims=true reshape form (matching
    // xmodel's xcompiler-side `shape_to_4d` output).  Keeping these as
    // ReduceMean lets that pass emit the cleaner `reshape -> reduce ->
    // reshape` instead of `transpose -> reshape -> transpose -> AvgPool`.
    // Keep the condition in sync with ReplaceQDQReductionPass.cpp.
    if (axes.size() == 1 && reduceOp.getKeepdims() != 0) {
      int64_t axis = axes[0];
      if ((rank == 4 && axis == 1) ||
          (rank == 3 && axis == 1 && inputShape[2] == 1))
        return failure();
    }

    // Only support spatial reduction for ReduceMean
    if (!areAxesValidForSpatialPooling(axes, rank)) {
      return failure();
    }

    auto [poolShape, kernelH, kernelW] =
        calculateSpatialPoolParameters(inputShape, axes);

    Value poolInput = reduceOp.getData();
    if (inputShape != ArrayRef<int64_t>(poolShape)) {
      poolInput = createReshapeOp(rewriter, loc, reduceOp.getData(), poolShape,
          inputType.getElementType());
    }

    auto kernelShapeAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto stridesAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto padsAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});

    // NCHW: poolShape = [N, C, H, W]
    int64_t outH = (poolShape[2] + kernelH - 1) / kernelH;
    int64_t outW = (poolShape[3] + kernelW - 1) / kernelW;
    SmallVector<int64_t> poolOutputShape = {
        poolShape[0], poolShape[1], outH, outW};
    auto poolOutputType =
        RankedTensorType::get(poolOutputShape, inputType.getElementType());

    // Create signed i64 type for ONNX attributes (si64)
    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);
    auto avgPoolOp =
        rewriter.create<ONNXAveragePoolOp>(loc, poolOutputType, poolInput,
            /*auto_pad=*/rewriter.getStringAttr("NOTSET"),
            /*ceil_mode=*/IntegerAttr::get(si64Type, 1),
            /*count_include_pad=*/IntegerAttr::get(si64Type, 0),
            /*dilations=*/nullptr, kernelShapeAttr, padsAttr, stridesAttr);

    Value result = avgPoolOp.getResult();

    if (poolOutputShape != outputType.getShape()) {
      result = createReshapeOp(rewriter, loc, result, outputType.getShape(),
          outputType.getElementType());
    }

    rewriter.replaceOp(reduceOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
/// Pattern: ReduceMax → MaxPool (Spatial Reduction)
//===----------------------------------------------------------------------===//

struct LowerReduceMaxToMaxPoolSpatialPattern
    : public OpRewritePattern<ONNXReduceMaxOp> {
  using OpRewritePattern<ONNXReduceMaxOp>::OpRewritePattern;

  /// match and rewrite the ReduceMax op to MaxPool op (spatial reduction)
  LogicalResult matchAndRewrite(
      ONNXReduceMaxOp reduceOp, PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();

    auto inputType = dyn_cast<RankedTensorType>(reduceOp.getData().getType());
    auto outputType = dyn_cast<RankedTensorType>(reduceOp.getType());

    if (!inputType || !outputType || !inputType.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    if (rank > 4 || rank < 3) {
      return failure();
    }

    // Extract axes
    auto axesOpt = extractAxesFromInput(reduceOp.getAxes());
    if (!axesOpt)
      return failure();

    auto axesResult = normalizeAndFilterAxes(*axesOpt, rank, inputShape,
        reduceOp, inputType, outputType, rewriter, loc);
    if (!axesResult)
      return success(); // Trivial reduction handled

    SmallVector<int64_t> axes = *axesResult;

    // This pattern handles SPATIAL reduction only
    if (!areAxesValidForSpatialPooling(axes, rank)) {
      return failure(); // Let channel pattern handle it
    }

    auto [poolShape, kernelH, kernelW] =
        calculateSpatialPoolParameters(inputShape, axes);

    Value poolInput = reduceOp.getData();
    if (inputShape != ArrayRef<int64_t>(poolShape)) {
      poolInput = createReshapeOp(rewriter, loc, reduceOp.getData(), poolShape,
          inputType.getElementType());
    }

    auto kernelShapeAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto stridesAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto padsAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});

    // NCHW: poolShape = [N, C, H, W]
    int64_t outH = (poolShape[2] + kernelH - 1) / kernelH;
    int64_t outW = (poolShape[3] + kernelW - 1) / kernelW;
    SmallVector<int64_t> poolOutputShape = {
        poolShape[0], poolShape[1], outH, outW};
    auto poolOutputType =
        RankedTensorType::get(poolOutputShape, inputType.getElementType());

    // Create signed i64 type for ONNX attributes (si64)
    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);

    // Create MaxPool (single output version)
    auto maxPoolOp =
        rewriter.create<ONNXMaxPoolSingleOutOp>(loc, poolOutputType, poolInput,
            /*auto_pad=*/rewriter.getStringAttr("NOTSET"),
            /*ceil_mode=*/IntegerAttr::get(si64Type, 1),
            /*dilations=*/nullptr,
            /*kernel_shape=*/kernelShapeAttr,
            /*pads=*/padsAttr,
            /*storage_order=*/IntegerAttr::get(si64Type, 0),
            /*strides=*/stridesAttr);

    Value result = maxPoolOp.getResult();

    if (poolOutputShape != outputType.getShape()) {
      result = createReshapeOp(rewriter, loc, result, outputType.getShape(),
          outputType.getElementType());
    }

    rewriter.replaceOp(reduceOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
/// Pattern: ReduceMax → MaxPool (Channel Reduction via Reshape Trick)
//===----------------------------------------------------------------------===//

struct LowerReduceMaxToMaxPoolChannelPattern
    : public OpRewritePattern<ONNXReduceMaxOp> {

  /// Lower priority than spatial pattern
  LowerReduceMaxToMaxPoolChannelPattern(MLIRContext *context)
      : OpRewritePattern<ONNXReduceMaxOp>(context, /*benefit=*/1) {}
  /// match and rewrite the ReduceMax op to MaxPool op (channel reduction via
  /// reshape trick)
  LogicalResult matchAndRewrite(
      ONNXReduceMaxOp reduceOp, PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();

    auto inputType = dyn_cast<RankedTensorType>(reduceOp.getData().getType());
    auto outputType = dyn_cast<RankedTensorType>(reduceOp.getType());

    if (!inputType || !outputType || !inputType.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    if (rank > 4 || rank < 3) {
      return failure();
    }

    // Extract axes
    auto axesOpt = extractAxesFromInput(reduceOp.getAxes());
    if (!axesOpt)
      return failure();

    SmallVector<int64_t> axes = *axesOpt;
    normalizeAxes(axes, rank);
    axes = removeTrivialAxes(axes, inputShape);

    if (axes.empty()) {
      return failure(); // Handled by spatial pattern
    }

    // This pattern handles reduction that INCLUDES channel dimension
    // or reduction on batch dimension (anything not pure spatial)
    if (areAxesValidForSpatialPooling(axes, rank)) {
      return failure(); // Let spatial pattern handle it
    }

    // Check if axes are continuous (required for reshape trick)
    if (!areAxesContinuous(axes)) {
      return failure(); // Non-continuous axes can't use reshape trick
    }

    // Calculate reshape parameters
    // Move reduction dimensions to channel position, then pool
    auto [poolShape, kernelH, kernelW] =
        calculateChannelPoolParameters(inputShape, axes);

    // Check kernel size limit
    if (kernelW > MAX_KERNEL_SIZE) {
      // Kernel too large - might need multi-pass or CPU fallback
      return failure();
    }

    /*
     * Transformation:
     *
     * Input: [N, H, W, C] with ReduceMax(axis=3)
     *   ↓
     * Reshape: [1, 1, N*H*W, C]  (move all non-reduced to W, reduced to C)
     *   ↓
     * MaxPool: kernel=[1, C], stride=[1, C]
     *   ↓
     * Output: [1, 1, N*H*W, 1]
     *   ↓
     * Reshape: [N, H, W, 1] (original output shape)
     */

    Value poolInput = createReshapeOp(rewriter, loc, reduceOp.getData(),
        poolShape, inputType.getElementType());

    auto kernelShapeAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto stridesAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto padsAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});

    // NCHW: Pool output after pooling across reduced dimension
    // poolShape = [1, 1, W, C] where C is the dimension being reduced
    int64_t outH = 1;
    int64_t outW = poolShape[2]; // W stays same
    SmallVector<int64_t> poolOutputShape = {1, 1, outH, outW};
    auto poolOutputType =
        RankedTensorType::get(poolOutputShape, inputType.getElementType());

    // Create signed i64 type for ONNX attributes (si64)
    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);

    auto maxPoolOp = rewriter.create<ONNXMaxPoolSingleOutOp>(loc,
        poolOutputType, poolInput, rewriter.getStringAttr("NOTSET"),
        IntegerAttr::get(si64Type, 0), nullptr, kernelShapeAttr, padsAttr,
        IntegerAttr::get(si64Type, 0), stridesAttr);

    Value result = maxPoolOp.getResult();

    // Reshape to expected output shape
    if (poolOutputShape != outputType.getShape()) {
      result = createReshapeOp(rewriter, loc, result, outputType.getShape(),
          outputType.getElementType());
    }

    rewriter.replaceOp(reduceOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
/// Pattern: ReduceSum → AveragePool + Mul(count)
//===----------------------------------------------------------------------===//

struct LowerReduceSumToAvgPoolPattern
    : public OpRewritePattern<ONNXReduceSumOp> {
  using OpRewritePattern<ONNXReduceSumOp>::OpRewritePattern;
  /// match and rewrite the ReduceSum op to AveragePool op + Mul(count)
  LogicalResult matchAndRewrite(
      ONNXReduceSumOp reduceOp, PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();

    auto inputType = dyn_cast<RankedTensorType>(reduceOp.getData().getType());
    auto outputType = dyn_cast<RankedTensorType>(reduceOp.getType());

    if (!inputType || !outputType || !inputType.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    if (rank > 4 || rank < 3) {
      return failure();
    }

    auto axesOpt = extractAxesFromInput(reduceOp.getAxes());
    if (!axesOpt)
      return failure();

    auto axesResult = normalizeAndFilterAxes(*axesOpt, rank, inputShape,
        reduceOp, inputType, outputType, rewriter, loc);
    if (!axesResult)
      return success(); // Trivial reduction handled

    SmallVector<int64_t> axes = *axesResult;

    // Defer to ReplaceQDQReductionPass for single-axis reductions it can
    // canonicalise to a rank-4 + keep_dims=true reshape form (matching
    // xmodel's xcompiler-side `shape_to_4d` output).  Keeping these as
    // ReduceSum lets that pass emit the cleaner `reshape -> reduce ->
    // reshape` instead of `transpose -> reshape -> transpose -> AvgPool ->
    // Mul(count)`.  Keep the condition in sync with
    // ReplaceQDQReductionPass.cpp.
    if (axes.size() == 1 && reduceOp.getKeepdims() != 0) {
      int64_t axis = axes[0];
      if ((rank == 4 && axis == 1) ||
          (rank == 3 && axis == 1 && inputShape[2] == 1))
        return failure();
    }

    if (!areAxesValidForSpatialPooling(axes, rank)) {
      return failure();
    }

    // Calculate reduction count
    int64_t reductionCount = 1;
    for (auto axis : axes) {
      reductionCount *= inputShape[axis];
    }

    auto [poolShape, kernelH, kernelW] =
        calculateSpatialPoolParameters(inputShape, axes);

    Value poolInput = reduceOp.getData();
    if (inputShape != ArrayRef<int64_t>(poolShape)) {
      poolInput = createReshapeOp(rewriter, loc, reduceOp.getData(), poolShape,
          inputType.getElementType());
    }

    auto kernelShapeAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto stridesAttr = rewriter.getI64ArrayAttr({kernelH, kernelW});
    auto padsAttr = rewriter.getI64ArrayAttr({0, 0, 0, 0});

    // NCHW: poolShape = [N, C, H, W]
    int64_t outH = (poolShape[2] + kernelH - 1) / kernelH;
    int64_t outW = (poolShape[3] + kernelW - 1) / kernelW;
    SmallVector<int64_t> poolOutputShape = {
        poolShape[0], poolShape[1], outH, outW};
    auto poolOutputType =
        RankedTensorType::get(poolOutputShape, inputType.getElementType());

    // Create signed i64 type for ONNX attributes (si64)
    auto si64Type =
        IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);
    auto avgPoolOp =
        rewriter.create<ONNXAveragePoolOp>(loc, poolOutputType, poolInput,
            /*auto_pad=*/rewriter.getStringAttr("NOTSET"),
            /*ceil_mode=*/IntegerAttr::get(si64Type, 1),
            /*count_include_pad=*/IntegerAttr::get(si64Type, 0),
            /*dilations=*/nullptr, kernelShapeAttr, padsAttr, stridesAttr);

    Value result = avgPoolOp.getResult();

    // Multiply by count to convert avg to sum
    // Handle float, integer, and quantized element types
    auto multiplier = static_cast<float>(reductionCount);
    Type elemType = inputType.getElementType();

    // Get storage type (for quantized types, this is the underlying int type)
    Type storageType = elemType;
    if (auto quantType = dyn_cast<quant::QuantizedType>(elemType))
      storageType = quantType.getStorageType();

    auto resultConstType = RankedTensorType::get({}, elemType);
    auto storageConstType = RankedTensorType::get({}, storageType);

    DenseElementsAttr multiplierAttr;
    if (isa<FloatType>(storageType)) {
      multiplierAttr = DenseElementsAttr::get(
          storageConstType, rewriter.getFloatAttr(storageType, multiplier));
    } else {
      // Integer/quantized: compute the quantized integer value
      int64_t intValue = static_cast<int64_t>(std::round(multiplier));
      if (auto uniformQType = dyn_cast<quant::UniformQuantizedType>(elemType)) {
        double scale = uniformQType.getScale();
        int64_t zp = uniformQType.getZeroPoint();
        intValue = static_cast<int64_t>(std::round(multiplier / scale)) + zp;
        intValue = std::max(
            intValue, static_cast<int64_t>(uniformQType.getStorageTypeMin()));
        intValue = std::min(
            intValue, static_cast<int64_t>(uniformQType.getStorageTypeMax()));
      }
      unsigned bitWidth = storageType.getIntOrFloatBitWidth();
      // Use isSigned=false to avoid APInt assertion; the bit pattern is what
      // matters, and DenseIntElementsAttr stores raw bits regardless of
      // signedness. Mask to bitWidth to ensure valid N-bit unsigned value.
      uint64_t maskedValue =
          static_cast<uint64_t>(intValue) & ((1ULL << bitWidth) - 1);
      multiplierAttr = DenseIntElementsAttr::get(
          storageConstType, ArrayRef<APInt>{APInt(bitWidth, maskedValue)});
    }

    auto multiplierConst = rewriter.create<ONNXConstantOp>(loc, resultConstType,
        Attribute(), multiplierAttr, FloatAttr(), ArrayAttr(), IntegerAttr(),
        ArrayAttr(), StringAttr(), ArrayAttr());

    auto mulOp = rewriter.create<ONNXMulOp>(
        loc, poolOutputType, result, multiplierConst);
    result = mulOp.getResult();

    if (poolOutputShape != outputType.getShape()) {
      result = createReshapeOp(rewriter, loc, result, outputType.getShape(),
          outputType.getElementType());
    }

    rewriter.replaceOp(reduceOp, result);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct LowerReduceToPoolPass
    : public PassWrapper<LowerReduceToPoolPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "lower-reduce-to-pool"; }
  StringRef getDescription() const override {
    return "Lower Reduce operations to Pool operations";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // ReduceMeanV13 → AveragePool (attribute-based axes; produced by
    // GlobalAveragePool canonicalization)
    patterns.add<LowerReduceMeanV13ToAvgPoolPattern>(context);
    // ReduceMean → AveragePool (operand-based axes)
    patterns.add<LowerReduceMeanToAvgPoolPattern>(context);
    // ReduceSum → AveragePool + Mul
    patterns.add<LowerReduceSumToAvgPoolPattern>(context);
    // ReduceMax → MaxPool (spatial) - higher priority
    patterns.add<LowerReduceMaxToMaxPoolSpatialPattern>(context);
    // ReduceMax → MaxPool (channel via reshape) - lower priority
    patterns.add<LowerReduceMaxToMaxPoolChannelPattern>(context);

    GreedyRewriteConfig config;
    config.maxIterations = 3;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createLowerReduceToPoolPass() {
  return std::make_unique<LowerReduceToPoolPass>();
}

} // namespace onnx_mlir
