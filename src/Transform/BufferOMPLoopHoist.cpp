#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_BUFFEROMPLOOPHOISTING
#include "src/Transform/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct BufferOMPLoopHoistingPass
    : public impl::BufferOMPLoopHoistingBase<BufferOMPLoopHoistingPass> {
  void runOnOperation() override;
};

bool directlyNestedIn(Operation *op, Operation *wsloopOp) {
  Operation *currentOp = op;
  Operation *parentOp;
  bool foundLoopNest =
      false; // flag for omp::LoopNestOp between alloc and wsloop
  while ((parentOp = currentOp->getParentOp()) != wsloopOp) {
    if (isa<omp::ParallelOp>(parentOp)) {
      // Need to worry other omp structure, such as task?
      return false;
    }
    if (isa<omp::LoopNestOp>(parentOp)) {
      foundLoopNest = true;
    }
    currentOp = parentOp;
  }

  return foundLoopNest;
}

bool areOperandsDefinedOutside(
    memref::AllocOp allocOp, omp::WsloopOp wsloopOp) {
  for (Value operand : allocOp.getOperands()) {
    Operation *currentOp = operand.getDefiningOp();
    while (Operation *parentOp = currentOp->getParentOp()) {
      if (parentOp == wsloopOp.getOperation())
        return false;
      currentOp = parentOp;
    }
  }
  return true;
}

std::optional<Operation *> findDealloc(
    memref::AllocOp allocOp, omp::WsloopOp wsloopOp) {
  // Find the only one dealloc. Otherwise nullOpt.
  std::optional<Operation *> deallocOpt =
      memref::findDealloc(allocOp.getResult());
  if (!deallocOpt)
    return deallocOpt;

  // Check whether the dealloc is in the wsloop and postdominate alloc
  Block *allocBlock = allocOp.getResult().getParentBlock();
  Block *deallocBlock = (*deallocOpt)->getBlock();
  allocBlock->dump();
  deallocBlock->dump();
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

  if (!deallocOpt)
    return deallocOpt;

  return deallocOpt;
}

// Hoist the alloc/dealloc in one omp.wsloop
void HandleOneLoop(omp::WsloopOp wsloopOp) {
  llvm::SmallVector<Operation *, 4> deallocList;
  llvm::SmallVector<Operation *, 4> allocList;
  wsloopOp.walk([&](memref::AllocOp allocOp) {
    std::optional<Operation *> deallocOpt = HandleOneAlloc(allocOp, wsloopOp);
    if (deallocOpt) {
      allocList.emplace_back(allocOp.getOperation());
      deallocList.emplace_back(*deallocOpt);
    }
  });

  // Perform hoisting
  for (auto alloc : allocList)
    alloc->moveBefore(wsloopOp.getOperation());
  for (auto dealloc : deallocList)
    dealloc->moveAfter(wsloopOp.getOperation());
}

void BufferOMPLoopHoistingPass::runOnOperation() {
  Operation *op = getOperation();
  op->walk([&](omp::WsloopOp wsloopOp) { HandleOneLoop(wsloopOp); });
}
} // namespace

namespace onnx_mlir {
std::unique_ptr<Pass> createBufferOMPLoopHoisting() {
  return std::make_unique<BufferOMPLoopHoistingPass>();
};
} // namespace onnx_mlir
