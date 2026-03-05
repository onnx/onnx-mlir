// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights reserved.

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

namespace onnx_mlir {

class ONNXCSEOperationInfo : public DenseMapInfo<Operation *> {
public:
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations |
            OperationEquivalence::IgnoreDiscardableAttrs);
  }

  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(const_cast<Operation *>(lhsC),
        const_cast<Operation *>(rhsC),
        OperationEquivalence::IgnoreLocations |
            OperationEquivalence::IgnoreDiscardableAttrs);
  }
};

class ONNXCSEPass
    : public PassWrapper<ONNXCSEPass, OperationPass<func::FuncOp>> {
public:
  [[nodiscard]] StringRef getName() const override { return "onnx-cse"; }

  [[nodiscard]] StringRef getArgument() const override { return "onnx-cse"; }

  [[nodiscard]] StringRef getDescription() const override {
    return "CSE that ignores discardable attributes, but only considers "
           "defined properties.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    llvm::SmallDenseSet<Operation *> opsToErase;
    // Don't erase ops while iterating with getOps
    for (Operation &op : func.getOps()) {
      if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (!memInterface.hasNoEffect())
          continue;
      }
      if (auto foundOpIter = knownOps.find(&op);
          foundOpIter != knownOps.end()) {
        op.replaceAllUsesWith((*foundOpIter)->getResults());
        opsToErase.insert(&op);
        continue;
      }
      knownOps.insert(&op);
    }
    for (auto *op : opsToErase)
      op->erase();
  }

private:
  DenseSet<Operation *, ONNXCSEOperationInfo> knownOps;
};

std::unique_ptr<Pass> createONNXCSEPass() {
  return std::make_unique<ONNXCSEPass>();
}

} // namespace onnx_mlir
