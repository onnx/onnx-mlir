/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to decompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the decomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

// Create an DenseElementsAttr of ArrayAttr.
// This function is used to get Value Type of an EXISTING ArrayAttr for Scaler
// function.
DenseElementsAttr createDenseArrayAttr(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {

  assert(origAttrs && "handle EXISTING ArrayAttr only");
  if (origAttrs.getValue()[0].dyn_cast<FloatAttr>()) {
    mlir::Type elementType = rewriter.getF32Type();
    int nElements = origAttrs.getValue().size();
    SmallVector<float, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i) {
      wrapper[i] = origAttrs.getValue()[i].cast<FloatAttr>().getValueAsDouble();
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }
  if (origAttrs.getValue()[0].dyn_cast<IntegerAttr>()) {
    mlir::Type elementType = rewriter.getIntegerType(64);
    int nElements = origAttrs.getValue().size();
    SmallVector<int64_t, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i) {
      wrapper[i] = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }
  llvm_unreachable("unexpected attribute type");
}

/// Create an Scalar DenseElementsAttr from FloatAttr or IntergerAttr.
/// This is used to create an ONNXConstant of rank 0, e.g. tensor<f32>.
DenseElementsAttr createScalarDenseAttr(
    PatternRewriter &rewriter, Attribute attr) {
  if (attr.dyn_cast<FloatAttr>()) {
    mlir::Type elementType = rewriter.getF32Type();
    SmallVector<float, 1> wrapper;
    wrapper.emplace_back(attr.cast<FloatAttr>().getValueAsDouble());
    return DenseElementsAttr::get(
        RankedTensorType::get({}, elementType), llvm::makeArrayRef(wrapper));
  }
  if (attr.dyn_cast<IntegerAttr>()) {
    mlir::Type elementType = rewriter.getIntegerType(64);
    SmallVector<int64_t, 1> wrapper;
    wrapper.emplace_back(attr.cast<IntegerAttr>().getInt());
    return DenseElementsAttr::get(
        RankedTensorType::get({}, elementType), llvm::makeArrayRef(wrapper));
  }
  llvm_unreachable("unexpected attribute type");
}

ConstantOp createUnitConstant(PatternRewriter &rewriter, Location loc) {
  return rewriter.create<ConstantOp>(loc, rewriter.getUnitAttr());
}

// Create an DenseElementsAttr of ArrayAttr.
// When ArrayAttr is Null, an empty Integer DenseElementAttr is returned
DenseElementsAttr createDenseArrayAttrOrEmpty(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {

  if (origAttrs) {
    return createDenseArrayAttr(rewriter, origAttrs);
  } else {
    mlir::Type elementType = rewriter.getIntegerType(64);
    int nElements = 0;
    SmallVector<int64_t, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i) {
      wrapper[i] = i;
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }
}

Value createSequenceConstructOp(
    PatternRewriter &rewriter, mlir::Value seq, mlir::OperandRange inputs) {
  Type resType = seq.getType();
  Location loc = seq.getLoc();
  Value position = rewriter.create<ConstantOp>(loc, rewriter.getUnitAttr());
  for (auto input : inputs) {
    seq = rewriter.create<ONNXSequenceInsertOp>(
        loc, resType, seq, input, position);
  }
  return seq;
}

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXDecompose.inc"

namespace {

struct DecomposeONNXToONNXPass
    : public PassWrapper<DecomposeONNXToONNXPass, OperationPass<FuncOp>> {

  StringRef getArgument() const override { return "decompose-onnx"; }

  StringRef getDescription() const override {
    return "Decompose ONNX operations into composition of other ONNX "
           "operations.";
  }

  void runOnOperation() final;
};
} // end anonymous namespace.

void DecomposeONNXToONNXPass::runOnOperation() {
  auto function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect, StandardOpsDialect>();

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  target.addIllegalOp<ONNXClipV6Op>();
  target.addIllegalOp<ONNXClipV11Op>();
  target.addIllegalOp<ONNXClipV12Op>();
  target.addIllegalOp<ONNXLogSoftmaxOp>();
  target.addIllegalOp<ONNXPadV2Op>();
  target.addIllegalOp<ONNXPadV11Op>();
  target.addIllegalOp<ONNXReduceL1Op>();
  target.addIllegalOp<ONNXReduceL2Op>();
  target.addIllegalOp<ONNXReduceLogSumOp>();
  target.addIllegalOp<ONNXReduceLogSumExpOp>();
  target.addIllegalOp<ONNXReduceSumSquareOp>();
  target.addIllegalOp<ONNXResizeV11Op>();
  target.addIllegalOp<ONNXResizeV10Op>();
  target.addIllegalOp<ONNXScalerOp>();
  target.addIllegalOp<ONNXSequenceConstructOp>();
  target.addIllegalOp<ONNXUpsampleOp>();
  target.addIllegalOp<ONNXUpsampleV9Op>();
  target.addIllegalOp<ONNXUpsampleV7Op>();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
} // end anonymous namespace

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createDecomposeONNXToONNXPass() {
  return std::make_unique<DecomposeONNXToONNXPass>();
}
