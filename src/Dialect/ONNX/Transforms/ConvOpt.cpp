/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ConvOpt.cpp - ONNX high level Convolution Optimizations ---------===//
//
// Copyright 2022-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of optimizations to optimize the execution of
// convolutions on CPUs.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXLayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ConvOpt.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

// Enables a minimum of printing.
#define DEBUG 0

using namespace mlir;

namespace onnx_mlir {
// Conv 1x1 to MatMul optimization has been moved to Decompose.cpp.
} // namespace onnx_mlir

namespace {

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Dialect/ONNX/Transforms/ONNXConvOpt.inc"

} // namespace

void onnx_mlir::getConvOptONNXToONNXPatterns(
    bool enableSimdDataLayoutOpt, RewritePatternSet &patterns) {
  // Conv 1x1 to MatMul optimization is now handled in Decompose pass.
  // This pass only handles SIMD data layout optimizations.
  if (enableSimdDataLayoutOpt)
    populateWithGenerated(patterns);
}

namespace {

struct ConvOptONNXToONNXPass
    : public PassWrapper<ConvOptONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvOptONNXToONNXPass)

  ConvOptONNXToONNXPass() = default;
  ConvOptONNXToONNXPass(const ConvOptONNXToONNXPass &pass)
      : mlir::PassWrapper<ConvOptONNXToONNXPass,
            OperationPass<func::FuncOp>>() {}
  ConvOptONNXToONNXPass(bool enableSimdDataLayout) {
    this->enableSimdDataLayoutOpt = enableSimdDataLayout;
  };

  StringRef getArgument() const override { return "conv-opt-onnx"; }

  StringRef getDescription() const override {
    return "Perform ONNX to ONNX optimizations for optimized CPU execution of "
           "convolutions.";
  }

  // Usage: onnx-mlir-opt --conv-opt-onnx='simd-data-layout'
  Option<bool> enableSimdDataLayoutOpt{*this, "simd-data-layout",
      llvm::cl::desc("Enable SIMD data layout optimizations"),
      ::llvm::cl::init(false)};

  void runOnOperation() final;
};

void ConvOptONNXToONNXPass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXDialect, arith::ArithDialect, func::FuncDialect>();

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  // Only add dynamic legality check if SIMD data layout optimization is
  // enabled.
  if (enableSimdDataLayoutOpt) {
    target.addDynamicallyLegalOp<ONNXConvOp>([&](ONNXConvOp op) {
      // Conv op has optimized layout
      bool hasOptLayout =
          onnx_mlir::hasConvONNXTensorDataLayout(op.getX().getType());
      if (DEBUG)
        fprintf(stderr,
            "ConvOps match&rewrite: went for the data simd layout opt.\n");
      if (hasOptLayout)
        assert(onnx_mlir::hasConvONNXTensorDataLayout(op.getW().getType()) &&
               "custom layout for both X and W");
      // Conv op is illegal (should be optimized) if it doesn't have optimized
      // layout.
      return hasOptLayout;
    });
  }

  RewritePatternSet patterns(context);
  onnx_mlir::getConvOptONNXToONNXPatterns(enableSimdDataLayoutOpt, patterns);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createConvOptONNXToONNXPass(
    bool enableSimdDataLayoutOpt) {
  return std::make_unique<ConvOptONNXToONNXPass>(enableSimdDataLayoutOpt);
}
