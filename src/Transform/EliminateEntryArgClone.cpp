/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- EliminateEntryArgClone.cpp - Drop clones of returned inputs ------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// The MLIR ownership-based buffer deallocation pass assumes that a function
// does not own its memref arguments. When such an argument (or a view of it) is
// returned, the pass inserts a `bufferization.clone` so that the caller
// receives a buffer it owns:
//
//   func.func @main_graph(%arg0: memref<?xf32>) -> memref<?xf32> {
//     %0 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
//     return %0 : memref<?xf32>
//   }
//
// For the entry function of an onnx-mlir model, this copy is unnecessary: the
// ownership of inputs and outputs is managed by the runtime. When converting
// Krnl to LLVM, an output that is traced back (through view-like ops) to an
// input argument is wrapped in an OMTensor that does not own its buffer (see
// shouldOwn in ConvertKrnlToLLVM.cpp).
//
// This pass removes such clones in functions referenced by krnl.entry_point,
// when the clone result is only used by func.return. It must run after
// bufferization::buildBufferDeallocationPipeline. Clones in other functions
// are kept, since their callers deallocate the returned buffers. Clones of
// returned krnl.global ops are kept as well, so that the output is writable.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "eliminate-entry-arg-clone"

using namespace mlir;

namespace {

/// Returns true when `v` is, possibly through view-like ops, an argument of the
/// entry block of `func`.
bool isFuncArg(Value v, func::FuncOp func) {
  while (Operation *defOp = v.getDefiningOp()) {
    auto viewOp = dyn_cast<ViewLikeOpInterface>(defOp);
    if (!viewOp)
      return false;
    v = viewOp.getViewSource();
  }
  auto arg = cast<BlockArgument>(v);
  return arg.getOwner() == &func.getBody().front();
}

class EliminateEntryArgClonePass
    : public PassWrapper<EliminateEntryArgClonePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateEntryArgClonePass)

  StringRef getArgument() const override { return "eliminate-entry-arg-clone"; }

  StringRef getDescription() const override {
    return "Remove bufferization.clone of entry function inputs that are "
           "directly returned.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp, 1> entryFuncs;
    module.walk([&](KrnlEntryPointOp entryPointOp) {
      auto funcRef = entryPointOp->getAttrOfType<SymbolRefAttr>(
          KrnlEntryPointOp::getEntryPointFuncAttrName());
      if (auto func = module.lookupSymbol<func::FuncOp>(funcRef))
        entryFuncs.emplace_back(func);
    });

    for (func::FuncOp func : entryFuncs) {
      SmallVector<bufferization::CloneOp> clones;
      func.walk([&](bufferization::CloneOp cloneOp) {
        Value result = cloneOp.getOutput();
        if (result.hasOneUse() &&
            isa<func::ReturnOp>(*result.getUsers().begin()) &&
            isFuncArg(cloneOp.getInput(), func))
          clones.emplace_back(cloneOp);
      });

      for (bufferization::CloneOp cloneOp : clones) {
        LLVM_DEBUG(llvm::dbgs() << "Removing " << cloneOp << "\n");
        Value input = cloneOp.getInput();
        // A clone may also cast its input, e.g. from a static to a dynamic
        // shape. Keep the result type with a memref.cast.
        if (input.getType() != cloneOp.getType()) {
          OpBuilder builder(cloneOp);
          input = memref::CastOp::create(
              builder, cloneOp.getLoc(), cloneOp.getType(), input);
        }
        cloneOp.getOutput().replaceAllUsesWith(input);
        cloneOp.erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> onnx_mlir::createEliminateEntryArgClonePass() {
  return std::make_unique<EliminateEntryArgClonePass>();
}
