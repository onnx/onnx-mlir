/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- EliminateWriteOnlyAlloc.cpp - Remove write-only local allocs ------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Eliminates locally-allocated memrefs that are only ever written to and never
// read from.  Such allocations arise when shape tensors for fused ops are
// lowered: the concat ops that computed the shapes are still present and
// generate store loops, but the fused-op lowering reads output *types* rather
// than runtime values, so those stores are dead.
//
// Detection uses MemoryEffectOpInterface so any store-like op (affine.store,
// krnl.store, memref.store, vector.store, krnl.memcpy-as-dest, …) is handled
// without enumerating specific types.  After erasing dead stores, the pass also
// cleans up any dead read-only ops (loads) that fed them, and removes
// affine.for loops whose bodies have become side-effect-free.
//
// Safety rule (fixpoint):
//   An alloc is safe to eliminate only when EVERY op that writes to it has ALL
//   of its other Write-target operands also in the safe-to-eliminate set.
//   This prevents erasing an op (e.g. zlow.lstm) that writes to both a dead
//   alloc and a live alloc: erasing the op would silently drop the live write.
//   The fixpoint iterates until no more candidates are pruned.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "eliminate-write-only-alloc"

using namespace mlir;

/// Returns true when every use of the alloc result only writes to or frees it
/// (no reads, no escapes to unknown code).  Uses MemoryEffectOpInterface so
/// any store-like dialect op is handled without enumerating specific types.
static bool isWriteOnlyAlloc(memref::AllocOp allocOp) {
  Value memRef = allocOp.getResult();
  for (Operation *user : memRef.getUsers()) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(user);
    if (!iface)
      return false; // Unknown effects — conservative
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffectsOnValue(memRef, effects);
    // If the op uses the memref but declares no effects on it (e.g. passes it
    // to an inner region), be conservative and keep the alloc.
    if (effects.empty())
      return false;
    // Any Read effect means the value is consumed — not write-only.
    for (auto &eff : effects)
      if (isa<MemoryEffects::Read>(eff.getEffect()))
        return false;
  }
  return true;
}

/// Returns true when any user of `allocVal` also writes to a memref that is
/// NOT in `candidates`.  Such a user cannot be safely erased (it has a live
/// write-target), so `allocVal` itself must survive.
static bool hasLiveCoWrite(
    Value allocVal, const llvm::DenseSet<Value> &candidates) {
  for (Operation *user : allocVal.getUsers()) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(user);
    if (!iface)
      // isWriteOnlyAlloc guarantees this branch is unreachable for any
      // candidate.  Return true conservatively so that if the assumption
      // ever breaks the alloc is pruned rather than silently mis-erased.
      return true;
    for (Value operand : user->getOperands()) {
      if (operand == allocVal)
        continue; // skip the candidate itself
      SmallVector<MemoryEffects::EffectInstance> effects;
      iface.getEffectsOnValue(operand, effects);
      for (auto &eff : effects)
        if (isa<MemoryEffects::Write>(eff.getEffect()) &&
            !candidates.count(operand))
          return true;
    }
  }
  return false;
}

/// Returns true when an affine.for loop has no SSA results and every op in its
/// body (besides the terminator) is either side-effect-free or has only Read
/// effects with unused results — i.e. the loop has no observable effect.
static bool isDeadAffineFor(affine::AffineForOp forOp) {
  if (!forOp.getResults().empty())
    return false;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!op.use_empty())
      return false;
    if (mlir::isMemoryEffectFree(&op))
      continue;
    // Allow ops that only read (safe to drop when result is unused).
    auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
    if (!iface)
      return false; // Unknown effects — conservative
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    for (auto &eff : effects)
      if (!isa<MemoryEffects::Read>(eff.getEffect()))
        return false; // Has Write/Allocate/Free — loop is not dead
  }
  return true;
}

/// Collect the stored value of any store-like op writing to `memRef`.
static void collectStoredValue(
    Operation *user, Value memRef, SmallVectorImpl<Value> &storedValues) {
  auto iface = dyn_cast<MemoryEffectOpInterface>(user);
  if (!iface)
    return;
  // The stored value is typically operand 0; find it as the operand that is
  // *not* the memref being written to.  For the common binary case (value,
  // memref) we just scan operands for non-memref values.
  // For well-known ops we use the typed accessor for clarity.
  if (auto op = dyn_cast<affine::AffineStoreOp>(user)) {
    storedValues.push_back(op.getValue());
  } else if (auto op = dyn_cast<KrnlStoreOp>(user)) {
    storedValues.push_back(op.getValue());
  } else if (auto op = dyn_cast<memref::StoreOp>(user)) {
    storedValues.push_back(op.getValue());
  }
  // Other store-like ops (vector.store, etc.) may not have a single scalar
  // "value" operand worth cleaning up — skip them here; the alloc erasure
  // is sufficient.
}

class EliminateWriteOnlyAllocPass
    : public PassWrapper<EliminateWriteOnlyAllocPass,
          OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override {
    return "eliminate-write-only-alloc";
  }

  StringRef getDescription() const override {
    return "Eliminate locally-allocated memrefs that are only written to.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Phase 1: Build the initial candidate set — allocs where every user only
    // has Write/Free effects on the alloc (no reads, no unknown ops).
    llvm::DenseSet<Value> candidates;
    func.walk([&](memref::AllocOp allocOp) {
      if (isWriteOnlyAlloc(allocOp))
        candidates.insert(allocOp.getResult());
    });

    LLVM_DEBUG(llvm::dbgs() << "[eliminate-write-only-alloc] "
                            << candidates.size() << " initial candidate(s)\n");

    // Phase 2: Fixpoint pruning.
    // Remove any candidate whose writer also writes to a live (non-candidate)
    // memref.  Erasing such a writer would silently drop a needed write.
    // Repeat until stable because pruning one candidate can expose another.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Value> toRemove;
      for (Value allocVal : candidates)
        if (hasLiveCoWrite(allocVal, candidates))
          toRemove.push_back(allocVal);
      for (Value v : toRemove) {
        LLVM_DEBUG({
          llvm::dbgs()
              << "[eliminate-write-only-alloc] pruned (live co-write): ";
          v.getDefiningOp<memref::AllocOp>().print(llvm::dbgs());
          llvm::dbgs() << "\n";
        });
        candidates.erase(v);
        changed = true;
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[eliminate-write-only-alloc] " << candidates.size()
               << " alloc(s) to eliminate after pruning\n");

    // Collect surviving candidates in program order for deterministic erasure.
    SmallVector<memref::AllocOp> toEliminate;
    func.walk([&](memref::AllocOp allocOp) {
      if (candidates.count(allocOp.getResult()))
        toEliminate.push_back(allocOp);
    });

    // Phase 3: Erase each dead alloc and its writers.
    for (memref::AllocOp allocOp : toEliminate) {
      Value memRef = allocOp.getResult();
      LLVM_DEBUG({
        llvm::dbgs() << "[eliminate-write-only-alloc] removing alloc: ";
        allocOp.print(llvm::dbgs());
        llvm::dbgs() << "\n";
        llvm::dbgs() << "  users to erase:\n";
        for (Operation *u : memRef.getUsers()) {
          llvm::dbgs() << "    ";
          u->print(llvm::dbgs());
          llvm::dbgs() << "\n";
        }
      });

      SmallVector<Value> storedValues;
      llvm::SmallSetVector<Operation *, 4> parentLoops;

      for (Operation *user : llvm::make_early_inc_range(memRef.getUsers())) {
        collectStoredValue(user, memRef, storedValues);

        Operation *parent = user->getParentOp();
        if (isa<affine::AffineForOp>(parent))
          parentLoops.insert(parent);

        user->erase();
      }

      allocOp.erase();

      // Erase any read-only op (load) that fed an erased store and is now
      // dead.  Using MemoryEffectOpInterface covers affine.load, krnl.load,
      // memref.load, and any other future load-like op.
      for (Value v : storedValues) {
        if (!v.use_empty())
          continue;
        Operation *defOp = v.getDefiningOp();
        if (!defOp)
          continue;
        if (mlir::isMemoryEffectFree(defOp)) {
          defOp->erase();
          continue;
        }
        auto iface = dyn_cast<MemoryEffectOpInterface>(defOp);
        if (!iface)
          continue;
        SmallVector<MemoryEffects::EffectInstance> effects;
        iface.getEffects(effects);
        bool onlyReads = llvm::all_of(effects,
            [](auto &e) { return isa<MemoryEffects::Read>(e.getEffect()); });
        if (onlyReads)
          defOp->erase();
      }

      // Erase parent affine.for loops that became fully side-effect-free.
      for (Operation *op : parentLoops)
        if (auto forOp = dyn_cast<affine::AffineForOp>(op))
          if (isDeadAffineFor(forOp))
            forOp.erase();
    }
  }
};

std::unique_ptr<Pass> onnx_mlir::createEliminateWriteOnlyAllocPass() {
  return std::make_unique<EliminateWriteOnlyAllocPass>();
}
