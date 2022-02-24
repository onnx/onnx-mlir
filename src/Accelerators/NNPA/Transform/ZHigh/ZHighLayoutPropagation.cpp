/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZHighLayoutPropagation.cpp - ZHigh High Level Optimizer ---===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "Dialect/ZHigh/ZHighHelper.hpp"
#include "Dialect/ZHigh/ZHighOps.hpp"
#include "Pass/DLCPasses.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ZHigh layout propagation Pass
//===----------------------------------------------------------------------===//

/// Get MemRef transposed by using permArray(d0, d1, d2, d3).
Value emitONNXTranspose(Location loc, PatternRewriter &rewriter, Value x,
    int d0, int d1, int d2, int d3) {
  ShapedType inputType = x.getType().cast<ShapedType>();
  Type elementType = inputType.getElementType();
  Type transposedType;
  if (inputType.hasRank()) {
    ArrayRef<int64_t> inputShape = inputType.getShape();
    SmallVector<int64_t, 4> transposedShape;
    transposedShape.emplace_back(inputShape[d0]);
    transposedShape.emplace_back(inputShape[d1]);
    transposedShape.emplace_back(inputShape[d2]);
    transposedShape.emplace_back(inputShape[d3]);
    transposedType = RankedTensorType::get(transposedShape, elementType);
  } else {
    transposedType = UnrankedTensorType::get(elementType);
  }

  SmallVector<int64_t, 4> permArray;
  permArray.emplace_back(d0);
  permArray.emplace_back(d1);
  permArray.emplace_back(d2);
  permArray.emplace_back(d3);
  ONNXTransposeOp transposedInput = rewriter.create<ONNXTransposeOp>(
      loc, transposedType, x, rewriter.getI64ArrayAttr(permArray));
  return transposedInput.getResult();
}

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Transform/ZHigh/ZHighLayoutPropagation.inc"

struct ZHighLayoutPropagationPass
    : public PassWrapper<ZHighLayoutPropagationPass, OperationPass<FuncOp>> {

  StringRef getArgument() const override { return "zhigh-layout-prop"; }

  StringRef getDescription() const override {
    return "Layout propagation at ZHighIR.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    // We want to canonicalize stick/unstick ops during this pass to simplify
    // rules in this pass.
    ZHighStickOp::getCanonicalizationPatterns(patterns, &getContext());
    ZHighUnstickOp::getCanonicalizationPatterns(patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};

} // end anonymous namespace.

std::unique_ptr<Pass> mlir::createZHighLayoutPropagationPass() {
  return std::make_unique<ZHighLayoutPropagationPass>();
}
