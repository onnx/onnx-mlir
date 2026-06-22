// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// XmcRequantizePass: post-quant-types analogue of
// `OptimizeOnnxRequantizationPass`. Inserts `XCOMPILERRequantize` on Group B
// data-flow ops whose operand and result `!quant.uniform` element types differ.
//
// Multi-use is handled here (unlike the original): post-quant-types, an
// op's mismatched (Q_a, Q_c) declared types are an IR inconsistency, so
// skipping multi-use cases would leave the backend with un-lowerable IR.
// We route all users through one shared Requantize via replaceAllUsesExcept
// (single-input ops) or per-operand setOperand (Concat).

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include <cmath>

using namespace mlir;

namespace {

static quant::UniformQuantizedType getPerTensorQuant(Value v) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  if (!t)
    return nullptr;
  return dyn_cast<quant::UniformQuantizedType>(t.getElementType());
}

static bool sameQuant(
    quant::UniformQuantizedType a, quant::UniformQuantizedType b) {
  return std::abs(a.getScale() - b.getScale()) < 1e-6f &&
         a.getZeroPoint() == b.getZeroPoint();
}

static ArrayAttr buildScaleAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedType q) {
  return rewriter.getArrayAttr(
      {rewriter.getF32FloatAttr(static_cast<float>(q.getScale()))});
}

static ArrayAttr buildZeroPointAttr(
    PatternRewriter &rewriter, quant::UniformQuantizedType q) {
  return rewriter.getI64ArrayAttr({q.getZeroPoint()});
}

// Single-input data-flow ops (kInputDictates): if op has (Q_a, Q_c) with
// Q_a != Q_c, retype result to Q_a and insert Requantize on the output edge.
template <typename OpType>
struct InputDictatesRequantizePattern : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      OpType op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumResults() < 1)
      return failure();

    Value operand = op->getOperand(0);
    Value result = op->getResult(0);

    auto inQ = getPerTensorQuant(operand);
    auto outQ = getPerTensorQuant(result);
    if (!inQ || !outQ)
      return failure();
    if (sameQuant(inQ, outQ))
      return failure();

    auto resultTy = cast<RankedTensorType>(result.getType());
    auto retypedTy = RankedTensorType::get(resultTy.getShape(), inQ);

    rewriter.modifyOpInPlace(op, [&]() { result.setType(retypedTy); });

    rewriter.setInsertionPointAfter(op);
    auto requant =
        rewriter.create<XCOMPILERRequantizeOp>(op.getLoc(), resultTy, result,
            buildScaleAttr(rewriter, inQ), buildZeroPointAttr(rewriter, inQ),
            buildScaleAttr(rewriter, outQ), buildZeroPointAttr(rewriter, outQ));

    rewriter.replaceAllUsesExcept(result, requant.getResult(), requant);
    return success();
  }
};

// Concat (kOutputDictates): for each operand whose quant differs from the
// result's quant, insert a Requantize between the producer and Concat.
struct ConcatRequantizePattern : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatOp op, PatternRewriter &rewriter) const override {
    Value result = op.getConcatResult();
    auto outQ = getPerTensorQuant(result);
    if (!outQ)
      return failure();

    bool anyChange = false;
    rewriter.setInsertionPoint(op);

    OperandRange inputs = op.getInputs();
    for (unsigned i = 0; i < inputs.size(); ++i) {
      Value operand = inputs[i];
      auto inQ = getPerTensorQuant(operand);
      if (!inQ)
        continue;
      if (sameQuant(inQ, outQ))
        continue;

      auto opType = cast<RankedTensorType>(operand.getType());
      auto newOperandTy = RankedTensorType::get(opType.getShape(), outQ);
      auto requant = rewriter.create<XCOMPILERRequantizeOp>(op.getLoc(),
          newOperandTy, operand, buildScaleAttr(rewriter, inQ),
          buildZeroPointAttr(rewriter, inQ), buildScaleAttr(rewriter, outQ),
          buildZeroPointAttr(rewriter, outQ));

      rewriter.modifyOpInPlace(
          op, [&]() { op->setOperand(i, requant.getResult()); });
      anyChange = true;
    }

    return anyChange ? success() : failure();
  }
};

} // namespace

namespace onnx_mlir {

struct XmcRequantizePass
    : public PassWrapper<XmcRequantizePass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "xmc-requantize"; }
  StringRef getDescription() const override {
    return "Insert XCOMPILERRequantize on mismatched-quant edges of "
           "Reshape/Transpose/Slice/Pad/DepthToSpace/SpaceToDepth/Concat/"
           "Squeeze/Unsqueeze/Flatten/Identity (post-quant-types "
           "analogue of OptimizeOnnxRequantizationPass).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<InputDictatesRequantizePattern<mlir::ONNXReshapeOp>,
        InputDictatesRequantizePattern<mlir::ONNXTransposeOp>,
        InputDictatesRequantizePattern<mlir::ONNXSliceOp>,
        InputDictatesRequantizePattern<mlir::ONNXPadOp>,
        InputDictatesRequantizePattern<mlir::ONNXDepthToSpaceOp>,
        InputDictatesRequantizePattern<mlir::ONNXSpaceToDepthOp>,
        InputDictatesRequantizePattern<mlir::ONNXSqueezeOp>,
        InputDictatesRequantizePattern<mlir::ONNXSqueezeV11Op>,
        InputDictatesRequantizePattern<mlir::ONNXUnsqueezeOp>,
        InputDictatesRequantizePattern<mlir::ONNXUnsqueezeV11Op>,
        InputDictatesRequantizePattern<mlir::ONNXFlattenOp>,
        InputDictatesRequantizePattern<mlir::ONNXIdentityOp>,
        ConcatRequantizePattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;

    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createXmcRequantizePass() {
  return std::make_unique<XmcRequantizePass>();
}

} // namespace onnx_mlir
