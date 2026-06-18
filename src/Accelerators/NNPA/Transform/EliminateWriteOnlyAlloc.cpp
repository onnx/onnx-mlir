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
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

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

    // Collect in a separate walk to avoid mutating the IR while iterating.
    SmallVector<memref::AllocOp> toEliminate;
    func.walk([&](memref::AllocOp allocOp) {
      if (isWriteOnlyAlloc(allocOp))
        toEliminate.push_back(allocOp);
    });

    for (memref::AllocOp allocOp : toEliminate) {
      Value memRef = allocOp.getResult();

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
