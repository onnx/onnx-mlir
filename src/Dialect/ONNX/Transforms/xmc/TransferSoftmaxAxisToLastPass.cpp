// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "transfer-softmax-axis-to-last"

using namespace mlir;

namespace {

struct TransferSoftmaxAxisToLastPattern
    : public OpRewritePattern<ONNXSoftmaxOp> {
  using OpRewritePattern<ONNXSoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSoftmaxOp op, PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !outputType)
      return failure();

    int64_t rank = inputType.getRank();
    if (rank == 0)
      return failure();

    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return failure();

    int64_t lastAxis = rank - 1;
    if (axis == lastAxis)
      return failure();

    if (!op.getResult().use_empty() &&
        llvm::all_of(op.getResult().getUsers(),
            [](Operation *user) { return isa<ONNXLogOp>(user); }))
      return failure();

    SmallVector<int64_t> orderIn;
    orderIn.reserve(rank);
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        orderIn.push_back(i);
    orderIn.push_back(axis);

    SmallVector<int64_t> orderOut(rank);
    for (int64_t i = 0; i < rank; ++i)
      orderOut[orderIn[i]] = i;

    Location loc = op.getLoc();
    SmallVector<int64_t> permutedShape(rank);
    for (int64_t i = 0; i < rank; ++i)
      permutedShape[i] = inputType.getShape()[orderIn[i]];

    Type permutedInputType =
        RankedTensorType::get(permutedShape, inputType.getElementType());
    auto inputTransposeOp = rewriter.create<ONNXTransposeOp>(loc,
        permutedInputType, op.getInput(), rewriter.getI64ArrayAttr(orderIn));

    Type permutedOutputType =
        RankedTensorType::get(permutedShape, outputType.getElementType());
    auto newSoftmaxOp = rewriter.create<ONNXSoftmaxOp>(loc, permutedOutputType,
        inputTransposeOp.getResult(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(64, /*isSigned=*/true), -1));

    auto outputTransposeOp =
        rewriter.create<ONNXTransposeOp>(loc, op.getResult().getType(),
            newSoftmaxOp.getResult(), rewriter.getI64ArrayAttr(orderOut));

    rewriter.replaceOp(op, outputTransposeOp.getResult());

    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct TransferSoftmaxAxisToLastPass
    : public PassWrapper<TransferSoftmaxAxisToLastPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-softmax-axis-to-last";
  }
  StringRef getDescription() const override {
    return "Wrap ONNX Softmax ops whose axis is not the last dimension with "
           "input/output Transposes that normalize the reduction axis to -1.";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<TransferSoftmaxAxisToLastPattern>(context);
    ResultNamesUpdater rnUpdater;
    GreedyRewriteConfig config;
    config.listener = &rnUpdater;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createTransferSoftmaxAxisToLastPass() {
  return std::make_unique<TransferSoftmaxAxisToLastPass>();
}

} // namespace onnx_mlir
