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
#include "Passes.hpp"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/onnx_decompose.inc"

struct DecomposeONNXToONNXPass : public FunctionPass<DecomposeONNXToONNXPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void DecomposeONNXToONNXPass::runOnFunction() {
  auto function = getFunction();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect>();

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  target.addIllegalOp<ONNXReduceL1Op>();
  target.addIllegalOp<ONNXReduceL2Op>();
  target.addIllegalOp<ONNXReduceLogSumOp>();
  target.addIllegalOp<ONNXReduceLogSumExpOp>();
  target.addIllegalOp<ONNXReduceSumSquareOp>();

  OwningRewritePatternList patterns;
  populateWithGenerated(context, &patterns);

  if (failed(applyPartialConversion(function, target, patterns)))
    signalPassFailure();
} // end anonymous namespace

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createDecomposeONNXToONNXPass() {
  return std::make_unique<DecomposeONNXToONNXPass>();
}

static PassRegistration<DecomposeONNXToONNXPass> pass("decompose-onnx",
    "Decompose ONNX operations into composition of other ONNX operations.");
