/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ProcessAffineParallelPrivate.cpp - handle parallel private data ---===//
//
// Copyright 2023-24 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ProcessAffineParallelPrivate.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "affine-parallel-private"

using namespace mlir;

void hi_super_alex() { fprintf(stderr, "hi super alex\n"); }
namespace {
func::FuncOp functionBeingDebugged;

struct ProcessAffineParallelWithoutScopePattern
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  static bool matchParallelForWithAllocScope(
      affine::AffineParallelOp parForOp) {
    fprintf(
        stderr, "\nhi alex, look for parallel for with/without alloca scope\n");
    parForOp.dump();
    // auto blockList = parForOp.getRegion().getBlocks();
    // if (!blockList || blockList->size() == 0) {
    if (parForOp.getRegion().empty()) {
      fprintf(stderr, "hi alex, ignore empty parallel region\n");
      return true;
    }
    Block *loopBody = parForOp.getBody();
    Operation &firstOp = loopBody->front();
    if (!isa<memref::AllocaScopeOp>(&firstOp)) {
      fprintf(
          stderr, "hi alex, found a parallel region without an alloca scope\n");
      return false;
    }
    fprintf(stderr, "hi alex, found a parallel region WITH an alloca scope\n");
    return true;
  }

  LogicalResult matchAndRewrite(affine::AffineParallelOp parForOp,
      PatternRewriter &rewriter) const final {
    fprintf(stderr, "hi alex, add alloca scope to a parallel region\n");
    Location loc = parForOp.getLoc();
    assert(!matchParallelForWithAllocScope(parForOp) &&
           "expected par for without alloca here");
#if 0
    // seems generally bad to clone the op as it has a hard time removing stuff later.
    auto newOp = rewriter.clone(*parForOp.getOperation());
    auto newParForOp = cast<affine::AffineParallelOp>(newOp);
#else
    SmallVector<Type, 4> resultTypes;
    for (auto t : parForOp.getResults()) {
      resultTypes.emplace_back(t.getType());
    }
    fprintf(stderr, "hi alex, there are %d results\n", (int)resultTypes.size());
    auto newParForOp = rewriter.create<affine::AffineParallelOp>(loc,
        resultTypes, parForOp.getReductionsAttr(), parForOp.getLowerBoundsMap(),
        parForOp.getLowerBoundsGroupsAttr(), parForOp.getUpperBoundsMap(),
        parForOp.getUpperBoundsGroupsAttr(), parForOp.getSteps(),
        parForOp.getMapOperands());
#if 0
    rewriter.inlineRegionBefore(parForOp.getRegion(), newParForOp.getRegion(),
        newParForOp.getRegion().begin());
#else
    newParForOp.getRegion().takeBody(parForOp.getRegion());
#endif
#endif
#if 1
    // Code inspired from SCFToOpenMP.cpp, in ParallelOpLowering struct, line
    // 399.
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
      fprintf(stderr, "\n\nhi alex before yield replace op\n");
#if 1 || 1
      auto newYield = rewriter.create<memref::AllocaScopeReturnOp>(
          loc, oldYield->getOperands());
      hi_super_alex();
      rewriter.replaceOp(oldYield, newYield);
#else
      rewriter.replaceOpWithNewOp<memref::AllocaScopeReturnOp>(
          oldYield, oldYield->getOperands());
#endif
      fprintf(stderr, "\n\nhi alex after yield replace op\n");
      fprintf(stderr, "in function\n");
      functionBeingDebugged.dump();
    }
#endif
    rewriter.replaceOp(parForOp, newParForOp);

    fprintf(stderr, "\n\nhi alex after replace parallel for op\n");
    newParForOp.dump();
    fprintf(stderr, "in function\n");
    functionBeingDebugged.dump();
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

  // hi alex
  functionBeingDebugged = function;

  fprintf(stderr, "hi alex, run process affine parallel private\n");

  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::vector::VectorDialect, mlir::scf::SCFDialect>();

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
  fprintf(stderr, "hi alex, done with parallel for alloca scope\n");
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
