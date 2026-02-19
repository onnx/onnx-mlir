// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;

namespace {

// Helper to check if a dimension range represents a full copy
static bool isFullDimCopy(int64_t begin, int64_t end, int64_t dimSize) {
  return begin == 0 && end == dimSize;
}

// Helper to create a constant tensor from int64 values
static Value createI64Constant(
    PatternRewriter &rewriter, Location loc, llvm::ArrayRef<int64_t> values) {
  onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
  return onnxBuilder.constantInt64(values);
}

// Helper to create a shape constant
static Value createShapeConstant(
    PatternRewriter &rewriter, Location loc, llvm::ArrayRef<int64_t> shape) {
  return createI64Constant(rewriter, loc, shape);
}

// Find the first of two consecutive dimensions that can be collapsed
// Returns dim0, where dim0 and dim0+1 are consecutive dimensions to collapse
static std::optional<int64_t> findDimsToCollapse(llvm::ArrayRef<int64_t> begins,
    llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> inputShape) {
  llvm::SmallVector<int64_t, 4> candidateDims;

  // Start from index 1 (skip batch dimension at 0)
  for (int64_t i = 1; i < static_cast<int64_t>(begins.size()); ++i) {
    if (isFullDimCopy(begins[i], ends[i], inputShape[i])) {
      if (!candidateDims.empty() && candidateDims.back() + 1 != i) {
        candidateDims.clear();
      }
      candidateDims.push_back(i);
      if (candidateDims.size() == 2) {
        return candidateDims[0];
      }
    }
  }
  return std::nullopt;
}

/// Pattern: transfer 5D quantized Slice operations to 4D by collapsing two
/// consecutive "full-copy" dimensions.
struct Transfer5dStridedSliceTo4dPattern
    : public OpRewritePattern<ONNXSliceOp> {
  using OpRewritePattern<ONNXSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSliceOp sliceOp, PatternRewriter &rewriter) const override {

    Value sliceInput = sliceOp.getData();
    auto inputType = dyn_cast<RankedTensorType>(sliceInput.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return failure();
    }

    // Get output tensor type
    auto outputType = dyn_cast<RankedTensorType>(sliceOp.getResult().getType());
    if (!outputType || !outputType.hasStaticShape()) {
      return failure();
    }

    // Check if input/output have quantized types
    auto inElemType = inputType.getElementType();
    auto outElemType = outputType.getElementType();

    auto inQType = dyn_cast<quant::UniformQuantizedType>(inElemType);
    auto outQType = dyn_cast<quant::UniformQuantizedType>(outElemType);

    // Both must be quantized and have matching quantization parameters
    if (!inQType || !outQType || inQType != outQType) {
      return failure();
    }

    // Note: Scale and zero-point are available via inQType.getScale() and
    // inQType.getZeroPoint() if needed. They are preserved in the quantized
    // type throughout the transformation.

    auto inputShape = inputType.getShape();

    // Check if input is 5D with batch dimension = 1
    if (inputShape.size() != 5 || inputShape[0] != 1) {
      return failure();
    }

    auto outputShape = outputType.getShape();
    if (outputShape.size() != 5 || outputShape[0] != 1) {
      return failure();
    }

    // Extract slice parameters from Slice operands
    llvm::SmallVector<int64_t, 5> begins;
    llvm::SmallVector<int64_t, 5> ends;
    llvm::SmallVector<int64_t, 5> strides;

    begins.reserve(5);
    ends.reserve(5);
    strides.reserve(5);

    // Lambda to extract int64 constant values from ONNX constant operations
    auto extractConstantInt64Values =
        [](Value val, llvm::SmallVectorImpl<int64_t> &result) -> bool {
      if (!val)
        return false;
      auto *defOp = val.getDefiningOp();
      if (!defOp)
        return false;

      if (defOp->getName().getStringRef() == "onnx.Constant") {
        if (auto valueAttr = defOp->getAttr("value")) {
          if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(valueAttr)) {
            result.reserve(result.size() + denseAttr.size());
            for (auto val : denseAttr.getValues<int64_t>()) {
              result.push_back(val);
            }
            return true;
          }
        }
      }
      return false;
    };

    extractConstantInt64Values(sliceOp.getStarts(), begins);
    extractConstantInt64Values(sliceOp.getEnds(), ends);
    if (sliceOp.getSteps()) {
      extractConstantInt64Values(sliceOp.getSteps(), strides);
    }

    if (begins.empty() || ends.empty()) {
      return failure();
    }

    if (strides.empty()) {
      strides.assign(begins.size(), 1);
    }

    // Find dimensions to collapse
    auto dimsToCollapse = findDimsToCollapse(begins, ends, inputShape);
    if (!dimsToCollapse) {
      return failure();
    }

    // dim0 is the first dimension to collapse
    // The second dimension (dim0 + 1) is implicitly consecutive
    int64_t dim0 = *dimsToCollapse;

    // Compute 4D shapes and parameters by collapsing dim0 and dim0+1
    llvm::SmallVector<int64_t, 4> input4DShape;
    llvm::SmallVector<int64_t, 4> output4DShape;
    llvm::SmallVector<int64_t, 4> begins4D;
    llvm::SmallVector<int64_t, 4> ends4D;
    llvm::SmallVector<int64_t, 4> strides4D;

    input4DShape.reserve(4);
    output4DShape.reserve(4);
    begins4D.reserve(4);
    ends4D.reserve(4);
    strides4D.reserve(4);

    // Build 4D arrays by collapsing dim0 and dim0+1
    for (int64_t i = 0; i < static_cast<int64_t>(inputShape.size()); ++i) {
      if (i == dim0) {
        input4DShape.push_back(inputShape[i] * inputShape[i + 1]);
        output4DShape.push_back(outputShape[i] * outputShape[i + 1]);
        begins4D.push_back(0);
        ends4D.push_back(input4DShape.back());
        strides4D.push_back(1);
        ++i; // Skip dim0+1 since we already processed it
      } else {
        input4DShape.push_back(inputShape[i]);
        output4DShape.push_back(outputShape[i]);
        begins4D.push_back(begins[i]);
        ends4D.push_back(ends[i]);
        strides4D.push_back(strides[i]);
      }
    }
    auto loc = sliceOp.getLoc();

    // Extract quantized element type (preserves quant.uniform type)
    auto quantElemType = inputType.getElementType();

    // Precompute types for 4D quantized tensors
    auto quant4DInputType = RankedTensorType::get(input4DShape, quantElemType);
    auto quant4DOutputType =
        RankedTensorType::get(output4DShape, quantElemType);

    // Create all constants upfront
    auto shapeConst4DInput = createShapeConstant(rewriter, loc, input4DShape);
    auto shapeConst5DOutput = createShapeConstant(rewriter, loc, outputShape);
    auto startsConst = createI64Constant(rewriter, loc, begins4D);
    auto endsConst = createI64Constant(rewriter, loc, ends4D);
    // axes4D is always [0, 1, 2, 3] for 4D tensors
    auto axesConst = createI64Constant(rewriter, loc, {0, 1, 2, 3});
    auto stepsConst = createI64Constant(rewriter, loc, strides4D);

    // Transform: Reshape(5D->4D) -> Slice4D -> Reshape(4D->5D)

    // Step 1: Reshape
    Value inputReshapeQuant4D =
        rewriter
            .create<ONNXReshapeOp>(
                loc, quant4DInputType, sliceInput, shapeConst4DInput)
            .getResult();

    // Step 2: Slice in 4D
    Value sliceQuant4D =
        rewriter
            .create<ONNXSliceOp>(loc, quant4DOutputType, inputReshapeQuant4D,
                startsConst, endsConst, axesConst, stepsConst)
            .getResult();

    // Step 3: Reshape back to 5D
    Value outputReshapeQuant5D = rewriter
                                     .create<ONNXReshapeOp>(loc, outputType,
                                         sliceQuant4D, shapeConst5DOutput)
                                     .getResult();

    rewriter.replaceOp(sliceOp, outputReshapeQuant5D);

    return success();
  }
};

} // namespace

namespace onnx_mlir {

/// Pass to transfer 5D quantized slice operations to 4D.
struct Transfer5dStridedSliceTo4dPass
    : public PassWrapper<Transfer5dStridedSliceTo4dPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-5d-strided-slice-to-4d";
  }
  StringRef getDescription() const override {
    return "Transfer 5D quantized Slice ops to equivalent 4D ops by collapsing "
           "two consecutive full-copy dimensions";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<Transfer5dStridedSliceTo4dPattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createTransfer5dStridedSliceTo4d() {
  return std::make_unique<Transfer5dStridedSliceTo4dPass>();
}

} // namespace onnx_mlir
