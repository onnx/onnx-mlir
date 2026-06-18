// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct BatchReductionToReshapeReductionPattern
    : public OpRewritePattern<ONNXReduceSumOp> {
  using OpRewritePattern<ONNXReduceSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReduceSumOp reduceOp, PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();

    // Match only quantized ReduceSum.
    Value inputValue = reduceOp.getData();
    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());
    if (!inputType)
      return failure();

    auto inputQType =
        dyn_cast<quant::UniformQuantizedType>(inputType.getElementType());
    if (!inputQType)
      return failure();

    if (inputType.getRank() != 4)
      return failure();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    if (inputShape[0] == 1)
      return failure();

    Value axesValue = reduceOp.getAxes();
    if (!axesValue)
      return failure();
    ElementsAttr axesAttr =
        onnx_mlir::getDenseOrDisposableConstLikeElements(axesValue);
    if (!axesAttr || !axesAttr.getElementType().isIntOrIndex())
      return failure();
    SmallVector<int64_t, 4> axesVec;
    for (APInt v : axesAttr.getValues<APInt>())
      axesVec.push_back(v.getSExtValue());
    if (axesVec.size() != 1)
      return failure();
    constexpr int64_t kInputRank = 4;
    int64_t canonicalInputAxis = axesVec[0];
    if (canonicalInputAxis < 0)
      canonicalInputAxis += kInputRank;
    if (canonicalInputAxis != 3)
      return failure();

    auto outputType =
        dyn_cast<RankedTensorType>(reduceOp.getResult().getType());
    if (!outputType)
      return failure();
    auto outputQType =
        dyn_cast<quant::UniformQuantizedType>(outputType.getElementType());
    if (!outputQType)
      return failure();

    // Shapes:
    //   preReshape  [1, dim0*dim1, dim2, dim3]
    //   reduceShape [1, dim0*dim1, dim2]      when keepdims = 0
    //               [1, dim0*dim1, dim2, 1]   when keepdims = 1
    //   postReshape original ReduceSum output shape
    int64_t dim0 = inputShape[0];
    int64_t dim1 = inputShape[1];
    int64_t dim2 = inputShape[2];
    int64_t dim3 = inputShape[3];
    int64_t flattenedDims = dim0 * dim1;

    // Inherit keepdims so the new ReduceSum's result rank stays consistent.
    int64_t origKeepdims = reduceOp.getKeepdims();

    SmallVector<int64_t, 4> preReshapeShape = {1, flattenedDims, dim2, dim3};

    // Normalize negative axis against the new (post-flatten) rank.
    int64_t reduceAxis = axesVec[0];
    if (reduceAxis < 0)
      reduceAxis += static_cast<int64_t>(preReshapeShape.size());

    // keepdims=1: keep reduced axis as size-1; keepdims=0: drop it.
    SmallVector<int64_t, 4> reduceShape(
        preReshapeShape.begin(), preReshapeShape.end());
    if (origKeepdims != 0)
      reduceShape[reduceAxis] = 1;
    else
      reduceShape.erase(reduceShape.begin() + reduceAxis);
    SmallVector<int64_t, 4> postReshapeShape(
        outputType.getShape().begin(), outputType.getShape().end());

    rewriter.setInsertionPoint(reduceOp);
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    Value preShapeConst = onnxBuilder.constantInt64(preReshapeShape);

    auto preReshapeType =
        RankedTensorType::get(preReshapeShape, inputType.getElementType());
    Value preReshape = rewriter
                           .create<ONNXReshapeOp>(
                               loc, preReshapeType, inputValue, preShapeConst)
                           .getResult();

    // Clone ReduceSum op generically to preserve attributes.
    Operation *oldReduce = reduceOp.getOperation();
    OperationState reduceState(loc, oldReduce->getName().getStringRef());
    reduceState.addTypes(
        RankedTensorType::get(reduceShape, outputType.getElementType()));

    SmallVector<Value, 4> newOperands;
    for (Value v : oldReduce->getOperands())
      newOperands.push_back(v);
    if (!newOperands.empty())
      newOperands[0] = preReshape;
    reduceState.addOperands(newOperands);
    for (NamedAttribute attr : oldReduce->getAttrs())
      reduceState.addAttribute(attr.getName(), attr.getValue());

    Operation *newReduce = rewriter.create(reduceState);

    rewriter.setInsertionPointAfter(newReduce);
    Value postShapeConst = onnxBuilder.constantInt64(postReshapeShape);
    Value postReshape = rewriter
                            .create<ONNXReshapeOp>(loc, outputType,
                                newReduce->getResult(0), postShapeConst)
                            .getResult();

    rewriter.replaceOp(reduceOp, postReshape);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct BatchReductionToReshapeReductionPass
    : public PassWrapper<BatchReductionToReshapeReductionPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "batch-reduction-to-reshape-reduction";
  }
  StringRef getDescription() const override {
    return "Convert batch ReduceSum on quantized tensors into "
           "reshape-optimized ReduceSum";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BatchReductionToReshapeReductionPattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createBatchReductionToReshapeReductionPass() {
  return std::make_unique<BatchReductionToReshapeReductionPass>();
}

} // namespace onnx_mlir
