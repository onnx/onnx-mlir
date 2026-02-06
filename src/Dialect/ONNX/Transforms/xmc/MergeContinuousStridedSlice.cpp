// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "merge-continuous-strided-slice"

using namespace mlir;

namespace {
/// Helper function to extract constant integer array from a tensor value
/// Returns failure if the value is not a constant or cannot be extracted
LogicalResult extractConstantIntArray(
    Value value, SmallVector<int32_t> &result) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return failure();

  // Try to match ONNXConstantOp
  if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(defOp)) {
    auto valueAttr = constOp.getValueAttr();
    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(valueAttr)) {
      result.clear();
      result.reserve(denseAttr.getNumElements());
      for (auto apInt : denseAttr.getValues<APInt>()) {
        result.push_back(apInt.getSExtValue());
      }
      return success();
    }
  }

  return failure();
}

/// Helper function to extract quantization parameters from a tensor type
/// Returns failure if the type is not quantized
LogicalResult extractQuantParamsFromType(
    Type type, double &scale, int64_t &zeroPoint) {
  auto tensorType = dyn_cast<TensorType>(type);
  if (!tensorType)
    return failure();

  auto quantType =
      dyn_cast<mlir::quant::UniformQuantizedType>(tensorType.getElementType());
  if (!quantType)
    return failure();

  scale = quantType.getScale();
  zeroPoint = quantType.getZeroPoint();
  return success();
}

/// Pattern to merge continuous chained Slice operations with quantized types:
///   %s1 = onnx.Slice(%x1, starts1, ends1, axes1, steps1) :
///   (tensor<...x!quant.uniform<u8:f32, scale:zp>>) ->
///   tensor<...x!quant.uniform<u8:f32, scale:zp>> %s2 = onnx.Slice(%s1,
///   starts2, ends2, axes2, steps2) : (tensor<...x!quant.uniform<u8:f32,
///   scale:zp>>) -> tensor<...x!quant.uniform<u8:f32, scale:zp>> %s3 =
///   onnx.Slice(%s2, starts3, ends3, axes3, steps3) :
///   (tensor<...x!quant.uniform<u8:f32, scale:zp>>) ->
///   tensor<...x!quant.uniform<u8:f32, scale:zp>>
/// becomes (if all scales and zero points are equal):
///   %s = onnx.Slice(%x1, merged_starts, merged_ends, merged_axes,
///   merged_steps) : (tensor<...x!quant.uniform<u8:f32, scale:zp>>) ->
///   tensor<...x!quant.uniform<u8:f32, scale:zp>>
struct MergeContinuousStridedSlicePattern
    : public OpRewritePattern<mlir::ONNXSliceOp> {
  using OpRewritePattern<mlir::ONNXSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::ONNXSliceOp sliceOp, PatternRewriter &rewriter) const override {
    DEBUG_WITH_TYPE("merge-continuous-strided-slice",
        llvm::errs() << "Trying to match " << sliceOp << "\n");

    // Step 1: Check if this slice has quantized input and output types
    Value inputVal = sliceOp.getData();
    Type inputType = inputVal.getType();
    Type outputType = sliceOp.getOutput().getType();

    double inputScale;
    double outputScale;
    int64_t inputZp;
    int64_t outputZp;

    if (failed(extractQuantParamsFromType(inputType, inputScale, inputZp)) ||
        failed(extractQuantParamsFromType(outputType, outputScale, outputZp))) {
      return rewriter.notifyMatchFailure(
          sliceOp, "Slice does not have quantized types");
    }

    // Verify input and output quantization parameters match
    if (inputScale != outputScale || inputZp != outputZp) {
      return rewriter.notifyMatchFailure(
          sliceOp, "Input and output quant params don't match");
    }

    double expectedScale = inputScale;
    int64_t expectedZp = inputZp;

    // Step 2: Build the chain of Slice operations
    SmallVector<mlir::ONNXSliceOp> sliceChain;
    sliceChain.push_back(sliceOp);

    // Build chain forwards (check if output feeds into another quantized slice)
    auto currentSlice = sliceOp;
    while (true) {
      Value sliceOutput = currentSlice.getOutput();
      if (!sliceOutput.hasOneUse()) {
        break;
      }

      auto nextSlice =
          dyn_cast<mlir::ONNXSliceOp>(*sliceOutput.getUsers().begin());
      if (!nextSlice) {
        break;
      }

      // Check if next slice has quantized types with matching parameters
      Type nextInputType = nextSlice.getData().getType();
      Type nextOutputType = nextSlice.getOutput().getType();

      double nextInputScale;
      double nextOutputScale;
      int64_t nextInputZp;
      int64_t nextOutputZp;

      if (failed(extractQuantParamsFromType(
              nextInputType, nextInputScale, nextInputZp)) ||
          failed(extractQuantParamsFromType(
              nextOutputType, nextOutputScale, nextOutputZp))) {
        break;
      }

      // Verify quantization parameters match
      if (nextInputScale != expectedScale || nextInputZp != expectedZp ||
          nextOutputScale != expectedScale || nextOutputZp != expectedZp) {
        break;
      }

      sliceChain.push_back(nextSlice);
      currentSlice = nextSlice;
    }

    // Build chain backwards (check if input comes from another quantized slice)
    currentSlice = sliceOp;
    while (true) {
      Value sliceInput = currentSlice.getData();
      auto prevSlice = sliceInput.getDefiningOp<mlir::ONNXSliceOp>();
      if (!prevSlice) {
        break;
      }

      // Check if prev slice has quantized types with matching parameters
      Type prevInputType = prevSlice.getData().getType();
      Type prevOutputType = prevSlice.getOutput().getType();

      double prevInputScale;
      double prevOutputScale;
      int64_t prevInputZp;
      int64_t prevOutputZp;

      if (failed(extractQuantParamsFromType(
              prevInputType, prevInputScale, prevInputZp)) ||
          failed(extractQuantParamsFromType(
              prevOutputType, prevOutputScale, prevOutputZp))) {
        break;
      }

      // Verify quantization parameters match
      if (prevInputScale != expectedScale || prevInputZp != expectedZp ||
          prevOutputScale != expectedScale || prevOutputZp != expectedZp) {
        break;
      }

      // Check if prev slice output has only one use (to current slice)
      if (!prevSlice.getOutput().hasOneUse()) {
        break;
      }

      sliceChain.insert(sliceChain.begin(), prevSlice);
      currentSlice = prevSlice;
    }

    if (sliceChain.size() < 2) {
      return rewriter.notifyMatchFailure(
          sliceOp, "Need at least 2 slices to merge");
    }

    auto headSlice = sliceChain[0];

    // Step 3: Extract constant arrays from all slices in chain
    SmallVector<SmallVector<int32_t>> startsVec;
    SmallVector<SmallVector<int32_t>> endsVec;
    SmallVector<SmallVector<int32_t>> axesVec;
    SmallVector<SmallVector<int32_t>> stepsVec;
    for (auto &op : sliceChain) {
      SmallVector<int32_t> starts;
      SmallVector<int32_t> ends;
      SmallVector<int32_t> axes;
      SmallVector<int32_t> steps;
      if (failed(extractConstantIntArray(op.getStarts(), starts)) ||
          failed(extractConstantIntArray(op.getEnds(), ends)) ||
          failed(extractConstantIntArray(op.getSteps(), steps))) {
        return rewriter.notifyMatchFailure(
            sliceOp, "Failed to extract constants");
      }
      // Handle axes - if it's None, assume all axes
      SmallVector<int32_t> axesValues;
      if (auto axesVal = op.getAxes()) {
        if (failed(extractConstantIntArray(axesVal, axesValues))) {
          return rewriter.notifyMatchFailure(sliceOp, "Failed to extract axes");
        }
      } else {
        // If axes is None, assume all dimensions
        // We'll use the size of starts/ends to determine dimensions
        for (int32_t i = 0; i < static_cast<int32_t>(starts.size()); ++i) {
          axesValues.push_back(i);
        }
      }

      startsVec.push_back(starts);
      endsVec.push_back(ends);
      axesVec.push_back(axesValues);
      stepsVec.push_back(steps);
    }

    // Step 4: Validate all slices have compatible dimensions
    size_t numDims = startsVec[0].size();
    for (size_t i = 1; i < startsVec.size(); ++i) {
      if (startsVec[i].size() != numDims || endsVec[i].size() != numDims ||
          stepsVec[i].size() != numDims || axesVec[i].size() != numDims) {
        return rewriter.notifyMatchFailure(sliceOp, "Incompatible dimensions");
      }
    }

    // Step 5: Compute merged starts, ends, steps
    auto newSteps = stepsVec[0];   // Initialize with head strides (steps)
    auto newStarts = startsVec[0]; // Initialize with head starts (begin)
    auto newEnds = endsVec[0];     // Initialize with head ends

    for (size_t i = 1; i < sliceChain.size(); ++i) {
      auto &steps = stepsVec[i];
      auto &starts = startsVec[i];
      auto &ends = endsVec[i];

      for (size_t j = 0; j < numDims; ++j) {
        newEnds[j] = newStarts[j] + newSteps[j] * ends[j];
        newStarts[j] += newSteps[j] * starts[j];
        newSteps[j] *= steps[j];
      }
    }

    // Step 6: Create new constant ops for merged parameters
    auto loc = sliceOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Create DenseIntElementsAttr for starts
    auto startsType =
        RankedTensorType::get({static_cast<int64_t>(newStarts.size())},
            IntegerType::get(ctx, 32, IntegerType::Signless));
    auto startsAttr = DenseIntElementsAttr::get(
        startsType, ArrayRef<int32_t>(newStarts.data(), newStarts.size()));

    // Create DenseIntElementsAttr for ends
    auto endsType =
        RankedTensorType::get({static_cast<int64_t>(newEnds.size())},
            IntegerType::get(ctx, 32, IntegerType::Signless));
    auto endsAttr = DenseIntElementsAttr::get(
        endsType, ArrayRef<int32_t>(newEnds.data(), newEnds.size()));

    // Create DenseIntElementsAttr for steps (strides)
    auto stepsType =
        RankedTensorType::get({static_cast<int64_t>(newSteps.size())},
            IntegerType::get(ctx, 32, IntegerType::Signless));
    auto stepsAttr = DenseIntElementsAttr::get(
        stepsType, ArrayRef<int32_t>(newSteps.data(), newSteps.size()));

    // Create axes (use the axes from the head slice)
    auto axesType =
        RankedTensorType::get({static_cast<int64_t>(axesVec[0].size())},
            IntegerType::get(ctx, 32, IntegerType::Signless));
    auto axesAttr = DenseIntElementsAttr::get(
        axesType, ArrayRef<int32_t>(axesVec[0].data(), axesVec[0].size()));

    // Create constant ops
    auto startsConst =
        rewriter.create<mlir::ONNXConstantOp>(loc, Attribute(), startsAttr);
    auto endsConst =
        rewriter.create<mlir::ONNXConstantOp>(loc, Attribute(), endsAttr);
    auto stepsConst =
        rewriter.create<mlir::ONNXConstantOp>(loc, Attribute(), stepsAttr);
    auto axesConst =
        rewriter.create<mlir::ONNXConstantOp>(loc, Attribute(), axesAttr);

    // Step 7: Create new merged Slice op with quantized types
    // Use the head slice's input and tail slice's output type
    Value headSliceInput = headSlice.getData();
    Type tailSliceOutputType = sliceChain.back().getOutput().getType();

    // Create new merged Slice op
    auto newSliceOp = rewriter.create<mlir::ONNXSliceOp>(loc,
        tailSliceOutputType, headSliceInput, startsConst.getOutput(),
        endsConst.getOutput(), axesConst.getOutput(), stepsConst.getOutput());

    // Step 8: Replace uses and erase old ops
    // Replace the tail slice's output with the new slice's output
    auto tailSlice = sliceChain.back();
    rewriter.replaceOp(tailSlice, newSliceOp.getOutput());

    // Erase all old slice ops (except tail which was replaced)
    for (size_t i = 0; i < sliceChain.size() - 1; ++i) {
      if (sliceChain[i]->use_empty()) {
        rewriter.eraseOp(sliceChain[i]);
      }
    }

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct MergeContinuousStridedSlicePass
    : public PassWrapper<MergeContinuousStridedSlicePass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "merge-continuous-strided-slice";
  }
  StringRef getDescription() const override {
    return "Merge continuous chained Slice operations with quantized types";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<MergeContinuousStridedSlicePattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createMergeContinuousStridedSlicePass() {
  return std::make_unique<MergeContinuousStridedSlicePass>();
}

} // namespace onnx_mlir
