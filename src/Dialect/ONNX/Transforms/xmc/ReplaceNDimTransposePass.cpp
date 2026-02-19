// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

/// Model-specific rewrite:
/// Match ONNXTranspose -> ONNXReshape, where perm == [2,0,3,1] and the
/// transpose output has static shape with dim1 == 1. Replace transpose with two
/// transposes: [0,2,1,3] then [0,1,3,2].
struct ReplaceNDimTransposePattern : public OpRewritePattern<ONNXTransposeOp> {
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXTransposeOp transposeOp, PatternRewriter &rewriter) const override {
    if (!transposeOp.getResult().hasOneUse())
      return failure();

    auto *userOp = *transposeOp.getResult().getUsers().begin();
    if (!dyn_cast<ONNXReshapeOp>(userOp))
      return failure();

    auto permAttr = transposeOp.getPermAttr();
    if (!permAttr)
      return failure();

    SmallVector<int64_t> permValues;
    for (auto attr : permAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      if (!intAttr)
        return failure();
      permValues.push_back(intAttr.getInt());
    }

    SmallVector<int64_t> targetPerm = {2, 0, 3, 1};
    if (permValues != targetPerm)
      return failure();

    auto outTy = dyn_cast<ShapedType>(transposeOp.getResult().getType());
    if (!outTy || !outTy.hasStaticShape())
      return failure();
    ArrayRef<int64_t> outShape = outTy.getShape();
    if (outShape.size() < 4 || outShape[1] != 1)
      return failure();

    Value transposeInput = transposeOp.getData();
    auto inTy = dyn_cast<ShapedType>(transposeInput.getType());
    if (!inTy || !inTy.hasStaticShape())
      return failure();
    if (inTy.getShape().size() < 4)
      return failure();

    SmallVector<int64_t> perm1 = {0, 2, 1, 3};
    auto perm1Attr = rewriter.getI64ArrayAttr(perm1);
    SmallVector<int64_t> shape1;
    for (auto idx : perm1)
      shape1.push_back(inTy.getShape()[idx]);
    auto outTy1 = RankedTensorType::get(shape1, inTy.getElementType());
    Value t1 = rewriter
                   .create<ONNXTransposeOp>(
                       transposeOp.getLoc(), outTy1, transposeInput, perm1Attr)
                   .getResult();

    SmallVector<int64_t> perm2 = {0, 1, 3, 2};
    auto perm2Attr = rewriter.getI64ArrayAttr(perm2);
    SmallVector<int64_t> shape2;
    for (auto idx : perm2)
      shape2.push_back(shape1[idx]);
    auto outTy2 = RankedTensorType::get(shape2, inTy.getElementType());
    Value t2 = rewriter
                   .create<ONNXTransposeOp>(
                       transposeOp.getLoc(), outTy2, t1, perm2Attr)
                   .getResult();

    rewriter.replaceOp(transposeOp, t2);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ReplaceNDimTransposePass : public PassWrapper<ReplaceNDimTransposePass,
                                      OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "replace-ndim-transpose"; }
  StringRef getDescription() const override {
    return "Model-specific transpose decomposition (replace-ndim-transpose)";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceNDimTransposePattern>(context);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createReplaceNDimTransposePass() {
  return std::make_unique<ReplaceNDimTransposePass>();
}

} // namespace onnx_mlir
