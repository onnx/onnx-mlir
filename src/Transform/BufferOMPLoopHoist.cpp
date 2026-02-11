#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "buffer-omploop-hoisting"

using namespace mlir;

namespace {

/* Include the definition of BufferOMPLoopHoistingBase from Passes.h.inc,
 * which is generated from Passes.td.
 * All the implementation of this pass is put in the anonymous name space
 * to hide from ourside.
 */
#define GEN_PASS_DEF_BUFFEROMPLOOPHOISTINGPASS
#include "src/Transform/Passes.h.inc"

struct BufferOMPLoopHoistingPass
    : public impl::BufferOMPLoopHoistingPassBase<BufferOMPLoopHoistingPass> {
  void runOnOperation() override;
};

bool directlyNestedIn(Operation *op, Operation *wsloopOp) {
  Operation *currentOp = op;
  Operation *parentOp;
  bool foundLoopNest =
      false; // flag for omp::LoopNestOp between alloc and wsloop
  while ((parentOp = currentOp->getParentOp()) != wsloopOp) {
    if (isa<omp::LoopNestOp>(parentOp)) {
      foundLoopNest = true;
    } else if (isa<LoopLikeOpInterface>(parentOp)) {
      // Can not be any other loops. This condition can be relaxed.
      return false;
    } else if (isa<omp::ParallelOp>(parentOp)) {
      // Need to worry other omp structure, such as task?
      return false;
    }
    currentOp = parentOp;
  }
  return foundLoopNest;
}

bool areOperandsDefinedOutside(
    memref::AllocOp allocOp, omp::WsloopOp wsloopOp) {
  for (Value operand : allocOp.getOperands()) {
    Operation *currentOp = operand.getDefiningOp();
    if (!currentOp)
      currentOp = mlir::cast<BlockArgument>(operand).getOwner()->getParentOp();
    while ((currentOp = currentOp->getParentOp())) {
      if (currentOp == wsloopOp.getOperation())
        return false;
    }
  }
  return true;
}

std::optional<Operation *> findDealloc(
    memref::AllocOp allocOp, omp::WsloopOp wsloopOp) {
  // Find the only one dealloc. Otherwise nullOpt.
  std::optional<Operation *> deallocOpt =
      memref::findDealloc(allocOp.getResult());
  if (!deallocOpt.has_value())
    return std::nullopt;
  // This check is needed due to the implementation of findDealloc.
  // When there is no use of the AllocOp, a std::optional<nullptr> is returned.
  if (!*deallocOpt)
    return std::nullopt;
  // Check whether the dealloc is in the wsloop and postdominate alloc
  Block *allocBlock = allocOp.getResult().getParentBlock();
  Block *deallocBlock = (*deallocOpt)->getBlock();
  // This is the simplified common case for us.
  if (allocBlock == deallocBlock)
    return deallocOpt;
  else
    return std::nullopt;
}

// Determine whether an memref::AllocOp can be hoisted out of the worksharing
// loop.
// Return std::optional of the corresponding dealloc if the alloc can be
// hoisted along with its dealloc
// Return std::nullopt if it can not be hoisted
std::optional<Operation *> HandleOneAlloc(
    memref::AllocOp allocOp, omp::WsloopOp wsloopOp) {
  // Check whether allocOp is directly in this wsloop, not in another
  // omp parallel structure.
  if (!directlyNestedIn(allocOp, wsloopOp))
    return std::nullopt;

  // Check the operands of allocOp is defined outside of wsloopOp
  // If not, the alloc can not be hoisted out.
  // It is assumed that loop invariants are already moved out of the loop.
  // No need to try back slicing here.
  if (!areOperandsDefinedOutside(allocOp, wsloopOp))
    return std::nullopt;

  // Check there is a dealloc for this alloc that makes sure the alloc
  // will not escape the wsloop body  so that the alloc/dealloc can be safely
  // hoisted out of wsloop.
  std::optional<Operation *> deallocOpt = findDealloc(allocOp, wsloopOp);

  if (!deallocOpt.has_value())
    return deallocOpt;

  return deallocOpt;
}

// Hoist the alloc/dealloc in an omp.WsloopOp
void HandleOneLoop(omp::WsloopOp wsloopOp) {
  llvm::SmallVector<Operation *, 4> deallocList;
  llvm::SmallVector<Operation *, 4> allocList;
  wsloopOp.walk([&](memref::AllocOp allocOp) {
    std::optional<Operation *> deallocOpt = HandleOneAlloc(allocOp, wsloopOp);
    if (deallocOpt.has_value()) {
      allocList.emplace_back(allocOp.getOperation());
      deallocList.emplace_back(*deallocOpt);
    }
  });

  // Perform hoisting
  for (auto alloc : allocList) {
    alloc->moveBefore(wsloopOp.getOperation());
    LLVM_DEBUG(llvm::dbgs() << "\nHoisted:\n");
    LLVM_DEBUG(alloc->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  for (auto dealloc : deallocList) {
    dealloc->moveAfter(wsloopOp.getOperation());
    LLVM_DEBUG(llvm::dbgs() << "\nHoisted:\n");
    LLVM_DEBUG(dealloc->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
}

void BufferOMPLoopHoistingPass::runOnOperation() {
  Operation *op = getOperation();
  op->walk([&](omp::WsloopOp wsloopOp) { HandleOneLoop(wsloopOp); });
}
} // namespace

namespace onnx_mlir {
#define GEN_PASS_DECL_BUFFEROMPLOOPHOISTINGPASS
#include "src/Transform/Passes.h.inc"

// This function will be used outside to insert this pass to pass manager.
// Since it is a pass in onnx-mlir project, name space onnx_mlir is used.
std::unique_ptr<Pass> createBufferOMPLoopHoisting() {
  return createBufferOMPLoopHoistingPass();
};
} // namespace onnx_mlir
