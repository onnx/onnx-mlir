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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighHelper.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace onnx_mlir {
namespace zhigh {

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
/// Use anonymous namespace to avoid duplication symbol `populateWithGenerated`
/// among multiple tablegen-based definitions.

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Transform/ZHigh/ONNXZHighLayoutPropagation.inc"

struct ZHighLayoutPropagationPass
    : public PassWrapper<ZHighLayoutPropagationPass,
          OperationPass<func::FuncOp>> {

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
} // anonymous namespace

std::unique_ptr<Pass> createZHighLayoutPropagationPass() {
  return std::make_unique<ZHighLayoutPropagationPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
