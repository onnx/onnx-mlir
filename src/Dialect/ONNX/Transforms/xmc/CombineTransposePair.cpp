// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "onnx-combine-transpose-pair"

using namespace mlir;

namespace {

/// Helper function to check if two perm attributes are equal
static bool arePermsEqual(ArrayAttr perm1, ArrayAttr perm2) {
  if (!perm1 || !perm2)
    return false;
  if (perm1.size() != perm2.size())
    return false;

  for (size_t i = 0; i < perm1.size(); ++i) {
    auto int1 = mlir::dyn_cast<IntegerAttr>(perm1[i]);
    auto int2 = mlir::dyn_cast<IntegerAttr>(perm2[i]);
    if (int1.getInt() != int2.getInt())
      return false;
  }
  return true;
}

/// Pattern to combine duplicate Transpose operations:
/// If two onnx.Transpose nodes have the same input and same perm,
/// keep only one and redirect all consumers.
///
/// Example:
///   %t1 = onnx.Transpose(%x) {perm = [0, 2, 1, 3]}
///   %t2 = onnx.Transpose(%x) {perm = [0, 2, 1, 3]}
///   use(%t1)
///   use(%t2)
/// becomes:
///   %t2 = onnx.Transpose(%x) {perm = [0, 2, 1, 3]}
///   use(%t2)
///   use(%t2)
struct CombineTransposePairPattern
    : public OpRewritePattern<ONNXTransposeOp> {
  using OpRewritePattern<ONNXTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXTransposeOp transposeOp,
      PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "combine-transpose-pair: Trying to match "
                            << transposeOp << "\n");

    Value input = transposeOp.getData();
    ArrayAttr perm = transposeOp.getPermAttr();

    // Find another Transpose op with the same input and perm
    for (Operation *user : input.getUsers()) {
      auto otherTranspose = dyn_cast<ONNXTransposeOp>(user);
      if (!otherTranspose || otherTranspose == transposeOp)
        continue;

      // Check if this transpose has the same perm
      ArrayAttr otherPerm = otherTranspose.getPermAttr();
      if (!arePermsEqual(perm, otherPerm))
        continue;

      // Replace all uses of the first transpose with the second one
      rewriter.replaceOp(transposeOp, otherTranspose.getResult());

      return success();
    }

    return rewriter.notifyMatchFailure(
        transposeOp, "No duplicate transpose found with same input and perm");
  }
};

} // namespace

namespace onnx_mlir {

/**
 * \brief Pass to combine duplicate ONNXTranspose operations.
 *
 * This pass identifies pairs of onnx.Transpose operations that have the same
 * input tensor and the same permutation attribute. Since these operations
 * produce identical results, the pass keeps one and redirects
 * all consumers to the remaining transpose.
 *
 * This optimization reduces redundant computation and memory usage when the
 * same transpose is computed multiple times in a model.
 */
struct CombineTransposePairPass
    : public PassWrapper<CombineTransposePairPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "combine-transpose-pair"; }
  StringRef getDescription() const override {
    return "Combine duplicate Transpose operations with same input and perm";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<CombineTransposePairPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createCombineTransposePairPass() {
  return std::make_unique<CombineTransposePairPass>();
}

} // namespace onnx_mlir
