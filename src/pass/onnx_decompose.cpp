//===- onnx_decompose.cpp - ONNX High Level Rewriting ---------------------===//
//
// Copyright 2019 The IBM Research Authors.
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

#include "src/dialect/onnx/onnx_ops.hpp"
#include "src/pass/passes.hpp"

using namespace mlir;

namespace {

// Check whether an ArrayAttr contains non-zero values or not.
bool hasNonZeroInArrayAttr(ArrayAttr attrs) {
  bool allZeros = true;
  if (attrs) {
    for (auto attr: attrs.getValue()) {
      if (attr.cast<IntegerAttr>().getInt() > 0) {
        allZeros = false;
        break;
      }
    }
  }
  return !allZeros;
}

// Create an ArrayAttr of IntergerAttr(s) of zero values.
// This function is used for padding attribute in MaxPoolSingleOut.
ArrayAttr createArrayAttrOfZeroWithTrail(
    PatternRewriter &rewriter, ArrayAttr origAttrs, int trailCount) {
  int nElements = origAttrs.getValue().size() + trailCount * 2;
  SmallVector<int64_t, 4> vals(nElements, 0);
  return rewriter.getI64ArrayAttr(vals);
}

// Pad a ArrayAttr with trails of zeros.
// This function is used for padding attribute in MaxPoolSingleOut.
ArrayAttr padArrayAttrWithZeroTrail(
    PatternRewriter &rewriter, ArrayAttr origAttrs, int trailCount) {
  int nDims = (int) origAttrs.getValue().size() / 2;
  int nElements = (nDims + trailCount) * 2;
  SmallVector<int64_t, 4> pads(nElements, 0);
  for (int i = 0; i < nDims; ++i) {
    int64_t beginPad = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();
    int64_t endPad =
        origAttrs.getValue()[nDims + i].cast<IntegerAttr>().getInt();
    pads[i + trailCount] = beginPad;
    pads[nDims + trailCount + i + trailCount] = endPad;
  }
  return rewriter.getI64ArrayAttr(pads);
}

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
  target.addIllegalOp<ONNXMaxPoolSingleOutOp>();

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
