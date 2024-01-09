/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ProcessAffineParallelPrivate.cpp - handle parallel private data ---===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
// This pass adds alloca_scope to parallel bodies to contain the memory
// allocated within its parallel body. Otherwise, temporary buffers would be
// shared among all threads.
//
// Input:
//   affine.parallel (%arg1) = (0) to (16384) step (32) {
//     body
//   }
//
// Output:
//   affine.parallel (%arg1) = (0) to (16384) step (32) {
//     memref.alloca_scope  {
//       body
//     }
//   }
//
// TODO: if we use scf.parallel, then the same optimization should be added as
// for the affine.parallel construct.
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

namespace {

struct ProcessAffineParallelWithoutScopePattern
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  static bool matchParallelForWithAllocScope(
      affine::AffineParallelOp parForOp) {
    if (parForOp.getRegion().empty()) {
      // Ignore empty parallel regions (side effects of optimization).
      return true;
    }
    Block *loopBody = parForOp.getBody();
    Operation &firstOp = loopBody->front();
    if (!isa<memref::AllocaScopeOp>(&firstOp)) {
      // Found a parallel region without an alloca scope, need to add one
      return false;
    }
    // Found a parallel region WITH an alloca scope, we are good.
    return true;
  }

  LogicalResult matchAndRewrite(affine::AffineParallelOp parForOp,
      PatternRewriter &rewriter) const final {
    Location loc = parForOp.getLoc();
    assert(!matchParallelForWithAllocScope(parForOp) &&
           "expected par for without alloca here");
    // Create a copy of the parallel for op, as this pass requires new ops.
    SmallVector<Type, 4> resultTypes;
    for (auto t : parForOp.getResults()) {
      resultTypes.emplace_back(t.getType());
    }
    auto newParForOp = rewriter.create<affine::AffineParallelOp>(loc,
        resultTypes, parForOp.getReductionsAttr(), parForOp.getLowerBoundsMap(),
        parForOp.getLowerBoundsGroupsAttr(), parForOp.getUpperBoundsMap(),
        parForOp.getUpperBoundsGroupsAttr(), parForOp.getSteps(),
        parForOp.getMapOperands());
    newParForOp.getRegion().takeBody(parForOp.getRegion());
    // Create a body that is surrounded by an alloca scope.
    // Code inspired from SCFToOpenMP.cpp, in ParallelOpLowering struct, line
    // 399.
    {
      OpBuilder::InsertionGuard allocaGuard(rewriter);
      // Create a block containing the ops in the loop body.
      Block *ops = rewriter.splitBlock(&*newParForOp.getRegion().begin(),
          newParForOp.getRegion().begin()->begin());
      auto oldYield = cast<affine::AffineYieldOp>(ops->getTerminator());

      // Insertion point at the top of the loop.
      rewriter.setInsertionPointToStart(&*newParForOp.getRegion().begin());
      // Create scope and affine yield.
      auto scope = rewriter.create<memref::AllocaScopeOp>(loc, TypeRange());
      rewriter.create<affine::AffineYieldOp>(loc, oldYield.getOperands());
      // Move the ops of the loop body into the alloca scope.
      Block *scopeBlock = rewriter.createBlock(&scope.getBodyRegion());
      rewriter.mergeBlocks(ops, scopeBlock);
      // Replace old yield by an alloca scope return.
      rewriter.setInsertionPointToEnd(&*scope.getBodyRegion().begin());
      rewriter.replaceOpWithNewOp<memref::AllocaScopeReturnOp>(
          oldYield, oldYield->getOperands());
    }
    rewriter.replaceOp(parForOp, newParForOp);
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

  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::vector::VectorDialect, mlir::scf::SCFDialect>();

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
