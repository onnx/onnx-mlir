/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ProcessKrnlParallelClause.cpp - handle Krnl Parallel Clauses ------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
// This pass seeks KrnlParallelClauseOp and integrate its parameter in the
// enclosing OpenMP Parallel construct.
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ProcessKrnlParallelClause.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "src/Support/TypeUtilities.hpp"

#define DEBUG_TYPE "krnl-parallel-clause"

using namespace mlir;

namespace {

struct ProcessKrnlParallelClauseWithoutScopePattern
    : public OpRewritePattern<KrnlParallelClauseOp> {
  using OpRewritePattern<KrnlParallelClauseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      KrnlParallelClauseOp clauseOp, PatternRewriter &rewriter) const final {
    // Get Parallel Krnl Clause
    Operation *op = clauseOp.getOperation();
    Value numThreads = clauseOp.getNumThreads();
    auto procBind = clauseOp.getProcBind();

    Operation *parentParallelOp = op->getParentOp();
    while (!llvm::dyn_cast_or_null<omp::ParallelOp>(parentParallelOp))
      parentParallelOp = parentParallelOp->getParentOp();

    if (parentParallelOp) {
      // Has an enclosing OpenMP parallel construct (expected).
      LLVM_DEBUG(llvm::dbgs()
                 << "Have a KrnlParallelClause with its OMP Parallel op\n");
      omp::ParallelOp parOp = llvm::cast<omp::ParallelOp>(parentParallelOp);
      if (numThreads) {
        LLVM_DEBUG(llvm::dbgs() << "  with a specific num_threads clause\n");
        // Set the numbers of threads as indicated by clause op.
        // WARNING: by moving the use of numThreads from inside the loop to the
        // outer OpenMP parallel construct, we may potentially move the use of
        // numThreads before its definition. However, because numThreads is by
        // definition loop invariant, it is very unlikely that this case occurs.
        // Nevertheless, this warning attests that this might be a possibility.
        // In such case, we would get a compiler warning/error of use before
        // def.
        MutableOperandRange mutableNumThreads = parOp.getNumThreadsMutable();
        mutableNumThreads.assign(numThreads);
      }
      if (procBind.has_value()) {
        auto str = procBind.value().str();
        LLVM_DEBUG(llvm::dbgs()
                   << "  with a specific proc_bind clause: " << str << "\n");
        // Set the affinity as indicated by the clause op.
        if (str == "primary")
          parOp.setProcBindKind(omp::ClauseProcBindKind::Primary);
        else if (str == "close")
          parOp.setProcBindKind(omp::ClauseProcBindKind::Close);
        else if (str == "spread")
          parOp.setProcBindKind(omp::ClauseProcBindKind::Spread);
        else
          llvm_unreachable("unkown proc_bind clause");
      }
    }
    // Useful info from KrnlParallelClauseOp was extracted, remove now.
    rewriter.eraseOp(op);
    return success();
  }
};

struct ProcessKrnlParallelClausePass
    : public PassWrapper<ProcessKrnlParallelClausePass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProcessKrnlParallelClausePass)

  ProcessKrnlParallelClausePass() {}
  ProcessKrnlParallelClausePass(const ProcessKrnlParallelClausePass &pass)
      : mlir::PassWrapper<ProcessKrnlParallelClausePass,
            OperationPass<func::FuncOp>>() {}

  StringRef getArgument() const override {
    return "process-krnl-parallel-clause";
  }

  StringRef getDescription() const override {
    return "Migrate info from Krnl Parallel Clause into OpenMP Parallel "
           "operation.";
  }

  void runOnOperation() final;

  typedef PassWrapper<ProcessKrnlParallelClausePass,
      OperationPass<func::FuncOp>>
      BaseType;
};

void ProcessKrnlParallelClausePass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::scf::SCFDialect, mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::vector::VectorDialect>();
  // Op that is used and removed here.
  target.addIllegalOp<KrnlParallelClauseOp>();

  RewritePatternSet patterns(context);
  onnx_mlir::getKrnlParallelClauseIntoOpenMPPatterns(patterns);

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

void onnx_mlir::getKrnlParallelClauseIntoOpenMPPatterns(
    mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<ProcessKrnlParallelClauseWithoutScopePattern>(context);
}

/*!
 * Create a Krnl Parallel Clause pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createProcessKrnlParallelClausePass() {
  return std::make_unique<ProcessKrnlParallelClausePass>();
}
