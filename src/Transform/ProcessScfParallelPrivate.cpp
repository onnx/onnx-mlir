/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ProcessScfParallelPrivate.cpp - handle parallel private data ---===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
// This pass adds alloca_scope to parallel bodies to contain the memory
// allocated within its parallel body. Otherwise, temporary buffers would be
// shared among all threads.
//
// Input:
//   scf.parallel (%arg1) = (0) to (16384) step (32) {
//     body
//   }
//
// Output:
//   scf.parallel (%arg1) = (0) to (16384) step (32) {
//     memref.alloca_scope  {
//       body
//     }
//   }
//
//===----------------------------------------------------------------------===//
#include "mlir/Transforms/Passes.h"

#include "src/Pass/Passes.hpp"
#include "src/Transform/ProcessScfParallelPrivate.hpp"

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

#define DEBUG_TYPE "scf-parallel-private"

using namespace mlir;

namespace {

/* All the implementation of this pass is put in the anonymous name space
 * to hide from ourside.
 */
#define GEN_PASS_DEF_PROCESSSCFPARALLELPRIVATEPASS
#include "src/Transform/Passes.h.inc"

struct ProcessScfParallelWithoutScopePattern
    : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  static bool matchParallelForWithAllocScope(scf::ParallelOp parForOp) {
    if (parForOp.getRegion().empty())
      // Ignore empty parallel regions (side effects of optimization).
      return true;
    Block *loopBody = parForOp.getBody();
    Operation &firstOp = loopBody->front();
    if (!isa<memref::AllocaScopeOp>(&firstOp))
      // Found a parallel region without an alloca scope, need to add one
      return false;
    // Found a parallel region WITH an alloca scope, we are good.
    return true;
  }

  LogicalResult matchAndRewrite(
      scf::ParallelOp parForOp, PatternRewriter &rewriter) const final {
    Location loc = parForOp.getLoc();
    assert(!matchParallelForWithAllocScope(parForOp) &&
           "expected par for without alloca here");
    auto newParForOp =
        scf::ParallelOp::create(rewriter, loc, parForOp.getLowerBound(),
            parForOp.getUpperBound(), parForOp.getStep(), parForOp.getInits());
    rewriter.eraseBlock(newParForOp.getBody());
    newParForOp.getRegion().takeBody(parForOp.getRegion());
    // Create a body that is surrounded by an alloca scope.
    // Code inspired from SCFToOpenMP.cpp, in ParallelOpLowering struct, line
    // 399.
    {
      OpBuilder::InsertionGuard allocaGuard(rewriter);
      // Create a block containing the ops in the loop body.
      Block *ops = rewriter.splitBlock(&*newParForOp.getRegion().begin(),
          newParForOp.getRegion().begin()->begin());
      auto oldYield = cast<scf::ReduceOp>(ops->getTerminator());
      // Insertion point at the top of the loop.
      rewriter.setInsertionPointToStart(&*newParForOp.getRegion().begin());
      // Create scope and scf yield.
      auto scope = memref::AllocaScopeOp::create(rewriter, loc, TypeRange());
      scf::ReduceOp::create(rewriter, loc, oldYield.getOperands());
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

struct ProcessScfParallelPrivatePass
    : public impl::ProcessScfParallelPrivatePassBase<
          ProcessScfParallelPrivatePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProcessScfParallelPrivatePass)

  void runOnOperation() final;
};

void ProcessScfParallelPrivatePass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::scf::SCFDialect, mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::vector::VectorDialect>();

  // Locate parallel for without scope
  target.addDynamicallyLegalOp<scf::ParallelOp>([](scf::ParallelOp op) {
    return ProcessScfParallelWithoutScopePattern::
        matchParallelForWithAllocScope(op);
  });
  RewritePatternSet patterns(context);
  onnx_mlir::getParallelPrivateScfToScfPatterns(patterns);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void onnx_mlir::getParallelPrivateScfToScfPatterns(
    mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<ProcessScfParallelWithoutScopePattern>(context);
}

/*!
 * Create a SCF Parallel Private pass.
 */
namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createProcessScfParallelPrivatePass() {
  return std::make_unique<ProcessScfParallelPrivatePass>();
}
} // namespace onnx_mlir
