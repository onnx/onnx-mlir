//===- QDQOpt.cpp - Remove QDQ operations --------*- C++ -*-===//
//
// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cmath>

using namespace mlir;
using namespace onnx_mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static ElementsAttr getElementAttributeFromConstant(Value val) {
  if (!val)
    return nullptr;
  if (auto constOp = val.getDefiningOp<ONNXConstantOp>())
    return mlir::dyn_cast<ElementsAttr>(constOp.getValueAttr());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Pattern to remove QDQ pairs
//===----------------------------------------------------------------------===//

struct FoldQDQPattern : public OpRewritePattern<ONNXQuantizeLinearOp> {
  using OpRewritePattern<ONNXQuantizeLinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXQuantizeLinearOp qOp, PatternRewriter &rewriter) const override {

    auto dqOp = qOp.getX().getDefiningOp<ONNXDequantizeLinearOp>();
    if (!dqOp)
      return failure();

    // 1. Check Attributes
    if (qOp.getAxis() != dqOp.getAxis())
      return failure();
    if (qOp.getBlockSize() != dqOp.getBlockSize())
      return failure();

    // 2. Check zero-points
    auto zpAttr1 = getElementAttributeFromConstant(dqOp.getXZeroPoint());
    auto zpAttr2 = getElementAttributeFromConstant(qOp.getYZeroPoint());
    if (!zpAttr1 && !zpAttr2)
      return failure();
    if (zpAttr1 != zpAttr2)
      return failure();

    // 3. Check Scales.
    auto scaleAttr1 = getElementAttributeFromConstant(dqOp.getXScale());
    auto scaleAttr2 = getElementAttributeFromConstant(qOp.getYScale());
    if (!scaleAttr1 && !scaleAttr2)
      return failure();
    if (scaleAttr1 != scaleAttr2)
      return failure();

    // 4. Check data type consistency of the entire DQ->Q chain.
    // The original quantized type before DQ must match the final quantized
    // type after Q.
    auto dqInTypeOp = dqOp.getX().getType();
    auto qOutTypeOp = qOp.getResult().getType();

    if (auto dqInTensorType = dqInTypeOp.dyn_cast<TensorType>()) {
      if (auto qOutTensorType = qOutTypeOp.dyn_cast<TensorType>()) {
        if (qOutTensorType.getElementType() !=
            dqInTensorType.getElementType()) {
          return failure();
        }
      } else {
        return failure();
      }
    } else {
      return failure();
    }
    rewriter.replaceOp(qOp, dqOp.getX());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass to run QDQ removal
//===----------------------------------------------------------------------===//

struct QDQOptONNXToONNXPass
    : public PassWrapper<QDQOptONNXToONNXPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QDQOptONNXToONNXPass)
  StringRef getArgument() const override { return "dqq-opt-onnx-to-onnx"; }
  StringRef getDescription() const override {
    return "Remove DqQ ops and surrounding DqQ if safe.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldQDQPattern>(&getContext());
    if (failed(applyPatternsGreedily(function, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createQDQOptONNXToONNXPass() {
  return std::make_unique<QDQOptONNXToONNXPass>();
}
} // namespace onnx_mlir