//
// Copyright (C) 2019 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

#include "llvm/ADT/SmallVector.h"

#include <numeric>
#include <optional>

using namespace mlir;

namespace {

//===- StandardizeSliceOpsPass.cpp - Standardize Slice Operations ---------===//

/// Check if a vector forms an arithmetic sequence
bool isArithmeticSequence(ArrayRef<int64_t> values) {
  if (values.size() < 2) {
    return true;
  }
  int64_t diff = values[1] - values[0];
  if (diff == 0) {
    return false;
  }
  for (size_t i = 2; i < values.size(); ++i) {
    if (values[i] - values[i - 1] != diff) {
      return false;
    }
  }
  return true;
}

/// Extract constant integer values from a DenseElementsAttr
std::optional<SmallVector<int64_t>> extractConstantIntegers(Value value) {
  if (!value) {
    return std::nullopt;
  }

  auto *defOp = value.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }

  // Check for onnx.Constant
  if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(defOp)) {
    if (auto valueAttr = constOp.getValueAttr()) {
      if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr)) {
        SmallVector<int64_t> result;
        for (auto val : denseAttr.getValues<APInt>()) {
          result.push_back(val.getSExtValue());
        }
        return result;
      }
    }
  }

  return std::nullopt;
}

/// Create a dense constant tensor
Value createConstantTensor(PatternRewriter &rewriter, Location loc,
    ArrayRef<int64_t> values, Type elementType) {
  auto tensorType =
      RankedTensorType::get({static_cast<int64_t>(values.size())}, elementType);
  auto denseAttr = DenseElementsAttr::get(tensorType, values);
  return rewriter.create<mlir::ONNXConstantOp>(loc, tensorType,
      mlir::ValueRange{},
      mlir::ArrayRef<mlir::NamedAttribute>{
          rewriter.getNamedAttr("value", denseAttr)});
}

/// Pattern 1: Standardize Slice Operations
struct StandardizeSlicePattern : public OpRewritePattern<mlir::ONNXSliceOp> {
  using OpRewritePattern<mlir::ONNXSliceOp>::OpRewritePattern;
  /// matchAndRewrite SliceOp
  LogicalResult matchAndRewrite(
      mlir::ONNXSliceOp sliceOp, PatternRewriter &rewriter) const override {
    Location loc = sliceOp.getLoc();

    // Get input shape
    auto inputType =
        mlir::dyn_cast<RankedTensorType>(sliceOp.getData().getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    // Extract starts, ends, axes, and steps
    auto starts = extractConstantIntegers(sliceOp.getStarts());
    auto ends = extractConstantIntegers(sliceOp.getEnds());

    if (!starts || !ends) {
      return failure(); // Can't standardize dynamic slices
    }

    // Get axes (default: all dimensions)
    SmallVector<int64_t> axes;
    if (sliceOp.getAxes() &&
        !mlir::isa<mlir::NoneType>(sliceOp.getAxes().getType())) {
      auto axesOpt = extractConstantIntegers(sliceOp.getAxes());
      if (axesOpt) {
        axes = *axesOpt;
      }
    } else {
      axes.resize(starts->size());
      std::iota(axes.begin(), axes.end(), 0);
    }

    // Get steps (default: 1 for all)
    SmallVector<int64_t> steps;
    if (sliceOp.getSteps() &&
        !mlir::isa<mlir::NoneType>(sliceOp.getSteps().getType())) {
      auto stepsOpt = extractConstantIntegers(sliceOp.getSteps());
      if (stepsOpt) {
        steps = *stepsOpt;
      }
    } else {
      steps.assign(starts->size(), 1);
    }

    // Create dense begin/end/strides for all dimensions
    SmallVector<int64_t> denseBegin(rank, 0);
    SmallVector<int64_t> denseEnd(inputShape.begin(), inputShape.end());
    SmallVector<int64_t> denseStrides(rank, 1);

    // Fill in the specified axes
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t axis = axes[i];
      if (axis < 0) {
        axis += rank;
      }

      int64_t start = (*starts)[i];
      int64_t end = (*ends)[i];
      int64_t step = steps[i];

      // Handle negative indices
      if (start < 0) {
        start += inputShape[axis];
      }
      if (end < 0) {
        end += inputShape[axis];
      }

      // Clamp to valid range
      start = std::max<int64_t>(0, std::min<int64_t>(start, inputShape[axis]));
      end = std::max<int64_t>(0, std::min<int64_t>(end, inputShape[axis]));

      denseBegin[axis] = start;
      denseEnd[axis] = end;
      denseStrides[axis] = step;
    }

    // Check if already standardized
    bool needsStandardization = (axes.size() != static_cast<size_t>(rank));
    if (!needsStandardization) {
      // Check if axes are in order 0, 1, 2, ...
      for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] != static_cast<int64_t>(i)) {
          needsStandardization = true;
          break;
        }
      }
    }

    if (!needsStandardization) {
      return failure(); // Already standardized
    }

    // Create new constant tensors for standardized attributes
    auto int64Type = rewriter.getI64Type();
    Value newStarts =
        createConstantTensor(rewriter, loc, denseBegin, int64Type);
    Value newEnds = createConstantTensor(rewriter, loc, denseEnd, int64Type);
    Value newSteps =
        createConstantTensor(rewriter, loc, denseStrides, int64Type);

    SmallVector<int64_t> allAxes(rank);
    std::iota(allAxes.begin(), allAxes.end(), 0);
    Value newAxes = createConstantTensor(rewriter, loc, allAxes, int64Type);

    // Create standardized Slice operation
    auto newSliceOp = rewriter.create<ONNXSliceOp>(loc, sliceOp.getType(),
        sliceOp.getData(), newStarts, newEnds, newAxes, newSteps);

    rewriter.replaceOp(sliceOp, newSliceOp.getOutput());

    return success();
  }
};

/// Pattern 2: Convert Gather with Arithmetic Sequence to Slice
struct ConvertGatherToSlicePattern
    : public OpRewritePattern<mlir::ONNXGatherOp> {
  using OpRewritePattern<mlir::ONNXGatherOp>::OpRewritePattern;
  /// matchAndRewrite GatherOp
  LogicalResult matchAndRewrite(
      mlir::ONNXGatherOp gatherOp, PatternRewriter &rewriter) const override {
    Location loc = gatherOp.getLoc();

    // Get input shape
    auto inputType =
        mlir::dyn_cast<RankedTensorType>(gatherOp.getData().getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    // Extract indices
    auto indices = extractConstantIntegers(gatherOp.getIndices());
    if (!indices || indices->empty()) {
      return failure();
    }

    // Check if indices form an arithmetic sequence
    if (!isArithmeticSequence(*indices)) {
      return failure();
    }

    // Get axis (default: 0)
    int64_t axis = gatherOp.getAxis();
    if (axis < 0) {
      axis += rank;
    }

    // Calculate stride
    int64_t stride =
        (indices->size() > 1) ? ((*indices)[1] - (*indices)[0]) : 1;

    // Calculate begin and end
    int64_t begin = (*indices)[0];
    int64_t end =
        (stride > 0) ? ((*indices).back() + 1) : ((*indices).back() - 1);

    // Create dense begin/end/strides for all dimensions
    SmallVector<int64_t> denseBegin(rank, 0);
    SmallVector<int64_t> denseEnd(inputShape.begin(), inputShape.end());
    SmallVector<int64_t> denseStrides(rank, 1);

    denseBegin[axis] = begin;
    denseEnd[axis] = end;
    denseStrides[axis] = stride;

    // Create constant tensors
    auto int64Type = rewriter.getI64Type();
    Value starts = createConstantTensor(rewriter, loc, denseBegin, int64Type);
    Value ends = createConstantTensor(rewriter, loc, denseEnd, int64Type);
    Value steps = createConstantTensor(rewriter, loc, denseStrides, int64Type);

    SmallVector<int64_t> allAxes(rank);
    std::iota(allAxes.begin(), allAxes.end(), 0);
    Value axes = createConstantTensor(rewriter, loc, allAxes, int64Type);

    // Compute the proper rank-preserving Slice output shape.
    // onnx.Slice always preserves rank, unlike onnx.Gather which can squeeze
    // the gather axis when the index is a scalar.
    SmallVector<int64_t> sliceShape;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t dimSize = (denseEnd[i] - denseBegin[i] + denseStrides[i] -
                            (denseStrides[i] > 0 ? 1 : -1)) /
                        denseStrides[i];
      sliceShape.push_back(std::max<int64_t>(0, dimSize));
    }

    auto sliceOutputType =
        RankedTensorType::get(sliceShape, inputType.getElementType());

    // Create Slice operation with the correct rank-preserving output type.
    auto sliceOp = rewriter.create<ONNXSliceOp>(
        loc, sliceOutputType, gatherOp.getData(), starts, ends, axes, steps);

    // If gather squeezed dimensions (scalar index), add a reshape to match
    // the original gather output shape.
    auto gatherOutputType = mlir::cast<RankedTensorType>(gatherOp.getType());

    if (sliceOutputType.getShape() != gatherOutputType.getShape()) {
      auto shapeType = RankedTensorType::get(
          {static_cast<int64_t>(gatherOutputType.getShape().size())},
          rewriter.getI64Type());
      auto shapeAttr =
          DenseElementsAttr::get(shapeType, gatherOutputType.getShape());
      auto shapeConst = rewriter.create<mlir::ONNXConstantOp>(loc, shapeType,
          mlir::ValueRange{},
          mlir::ArrayRef<mlir::NamedAttribute>{
              rewriter.getNamedAttr("value", shapeAttr)});

      auto reshapeOp = rewriter.create<ONNXReshapeOp>(loc, gatherOutputType,
          sliceOp.getResult(), shapeConst,
          /*allowzero=*/0);
      rewriter.replaceOp(gatherOp, reshapeOp.getResult());
    } else {
      rewriter.replaceOp(gatherOp, sliceOp.getOutput());
    }

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct StandardizeSliceOpsPass
    : public PassWrapper<StandardizeSliceOpsPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "standardize-slice-ops"; }
  StringRef getDescription() const override {
    return "Standardize Slice operations and convert Gather to Slice";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<StandardizeSlicePattern>(context);
    patterns.add<ConvertGatherToSlicePattern>(context);

    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createStandardizeSliceOpsPass() {
  return std::make_unique<StandardizeSliceOpsPass>();
}

} // namespace onnx_mlir
