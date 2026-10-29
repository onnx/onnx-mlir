/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OMPWsloopNowait.cpp - Drop redundant wsloop barriers ---------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass that marks the last omp.wsloop
// of an omp.parallel region as nowait when only per-thread code separates it
// from the end of the region. The implicit barrier of such a loop is
// redundant with the join at the end of the parallel region.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "omp-wsloop-nowait"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_OMPWSLOOPNOWAITPASS
#include "src/Transform/Passes.h.inc"
} // namespace onnx_mlir

namespace {

// Return true when op, located after the wsloop in the body block of
// parallelOp, only performs work that is private to the executing thread.
bool isPerThreadTailOp(Operation *op, omp::ParallelOp parallelOp) {
  // Nested regions could hide arbitrary synchronization; be conservative.
  if (op->getNumRegions() != 0)
    return false;
  if (isMemoryEffectFree(op))
    return true;
  // Dealloc of a buffer that each thread allocated itself in the region (as
  // created by the buffer-omploop-hoisting pass).
  if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
    auto allocOp = deallocOp.getMemref().getDefiningOp<memref::AllocOp>();
    return allocOp && allocOp->getBlock() == &parallelOp.getRegion().front();
  }
  return false;
}

// Return true when the implicit barrier at the end of wsloopOp can be removed.
bool canBeNowait(omp::WsloopOp wsloopOp) {
  if (wsloopOp.getNowait())
    return false;
  // Clauses whose semantics may rely on the barrier at the end of the loop.
  if (wsloopOp.getNumReductionVars() > 0 ||
      !wsloopOp.getPrivateVars().empty() || wsloopOp.getPrivateNeedsBarrier() ||
      !wsloopOp.getLinearVars().empty())
    return false;
  // Must be directly in a single-block parallel region.
  auto parallelOp = dyn_cast<omp::ParallelOp>(wsloopOp->getParentOp());
  if (!parallelOp || !parallelOp.getRegion().hasOneBlock())
    return false;
  // Every op between the wsloop and the terminator must be per-thread.
  for (Operation *op = wsloopOp->getNextNode(); op; op = op->getNextNode()) {
    if (isa<omp::TerminatorOp>(op))
      return op == &parallelOp.getRegion().front().back();
    if (!isPerThreadTailOp(op, parallelOp))
      return false;
  }
  return false;
}

class OMPWsloopNowaitPass
    : public onnx_mlir::impl::OMPWsloopNowaitPassBase<OMPWsloopNowaitPass> {
public:
  void runOnOperation() override {
    getOperation()->walk([&](omp::WsloopOp wsloopOp) {
      if (canBeNowait(wsloopOp)) {
        wsloopOp.setNowait(true);
        LLVM_DEBUG(llvm::dbgs() << "\nSet nowait:\n");
        LLVM_DEBUG(wsloopOp->print(llvm::dbgs()));
        LLVM_DEBUG(llvm::dbgs() << "\n");
      }
    });
  }
};

} // namespace
