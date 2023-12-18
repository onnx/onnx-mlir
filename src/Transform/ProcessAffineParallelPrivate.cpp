/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXRecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to Recompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the Recomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point.
//
// TODO: This file is quite busy as the number of decomposing op is increasing.
// It is better to move decomposition of each operation into a separate file.
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ProcessAffineParallelPrivate.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "affine-parallel-private"

using namespace mlir;

namespace {

struct ProcessAffineParallelWithoutScopePattern
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  static bool matchParallelForWithAllocScope(
      affine::AffineParallelOp parForOp) {
    Block *loopBody = parForOp.getBody();
    Operation &firstOp = loopBody->front();
    if (!isa<memref::AllocaScopeOp>(&firstOp)) {
      fprintf(
          stderr, "hi alex, found a parallel region without an alloca scope\n");
      parForOp->dump();
      return false;
    }
    fprintf(stderr, "hi alex, found a parallel region WITH an alloca scope\n");
    parForOp->dump();
    return true;
  }

  LogicalResult matchAndRewrite(affine::AffineParallelOp parForOp,
      PatternRewriter &rewriter) const final {
    fprintf(stderr, "hi alex, add alloca scope to a parallel region\n");
    Location loc = parForOp.getLoc();
    assert(!matchParallelForWithAllocScope(parForOp) &&
           "expected par for without alloca here");
#if 1
    auto newOp = rewriter.clone(*parForOp.getOperation());
    auto newParForOp = cast<affine::AffineParallelOp>(newOp);
#else
    auto newParForOp = rewriter.create<affine::AffineParallelOp>(loc,
        parForOp.getReductionsAttr(), parForOp.getLowerBoundsMapAttr(),
        parForOp.getLowerBoundsGroupsAttr(), parForOp.getUpperBoundsMapAttr(),
        parForOp.getUpperBoundsGroupsAttr(), parForOp.getSteps());
    newParForOp.getRegion().takeBody(parForOp.getRegion());
#endif
    {
      OpBuilder::InsertionGuard allocaGuard(rewriter);
      // Create a block containing the ops in the loop body.
      Block *ops = rewriter.splitBlock(&*newParForOp.getRegion().begin(),
          newParForOp.getRegion().begin()->begin());
      // Insertion point at the top of the loop.
      rewriter.setInsertionPointToStart(&*newParForOp.getRegion().begin());
      // Create scope and affine yield.
      auto scope = rewriter.create<memref::AllocaScopeOp>(loc, TypeRange());
      auto parForYieldOp =
          rewriter.create<affine::AffineYieldOp>(loc, ValueRange());

      // Move the ops of the loop body into the alloca scope.
      Block *scopeBlock = rewriter.createBlock(&scope.getBodyRegion());
      rewriter.mergeBlocks(ops, scopeBlock);

      auto oldYield = cast<affine::AffineYieldOp>(scopeBlock->getTerminator());
      // parForYieldOp.setOperand(oldYield->getOperand());
      rewriter.setInsertionPointToEnd(&*scope.getBodyRegion().begin());
      fprintf(stderr, "hi alex before replace op\n");
      //rewriter.replaceOpWithNewOp<memref::AllocaScopeReturnOp>(
      //    oldYield, oldYield->getOperands());
      //fprintf(stderr, "hi alex after replace op\n");
    }
    rewriter.replaceOp(parForOp, newParForOp);

    fprintf(stderr, "hi alex after replace parallel for op\n");
    newParForOp.dump();
    return success();
  }
};

struct ProcessAffineParallelPrivatePass
    : public PassWrapper<ProcessAffineParallelPrivatePass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProcessAffineParallelPrivatePass)

  ProcessAffineParallelPrivatePass() {}
  ProcessAffineParallelPrivatePass(const ProcessAffineParallelPrivatePass &pass)
      : mlir::PassWrapper<ProcessAffineParallelPrivatePass,
            OperationPass<func::FuncOp>>() {}

  StringRef getArgument() const override { return "affine-parallel-private"; }

  StringRef getDescription() const override {
    return "Process affine parallel for op to support private variables.";
  }

  void runOnOperation() final;

  typedef PassWrapper<ProcessAffineParallelPrivatePass,
      OperationPass<func::FuncOp>>
      BaseType;
};

void ProcessAffineParallelPrivatePass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  fprintf(stderr, "hi alex, run process affine parallel private\n");

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();

#if 1
  // Locate parallel for without scope
  target.addDynamicallyLegalOp<affine::AffineParallelOp>(
      [](affine::AffineParallelOp op) {
        return ProcessAffineParallelWithoutScopePattern::
            matchParallelForWithAllocScope(op);
      });

  RewritePatternSet patterns(context);
  onnx_mlir::getParallelPrivateAffineToAffinePatterns(patterns);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
#endif
}

} // namespace

void onnx_mlir::getParallelPrivateAffineToAffinePatterns(
    mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<ProcessAffineParallelWithoutScopePattern>(context);
}

/*!
 * Create a RecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass>
onnx_mlir::createProcessAffineParallelPrivatePass() {
  return std::make_unique<ProcessAffineParallelPrivatePass>();
}
