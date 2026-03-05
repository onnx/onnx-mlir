// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights reserved.

#include "mlir/IR/Attributes.h"
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
    auto *op = const_cast<Operation *>(opC);
    llvm::hash_code hash = llvm::hash_combine(op->getName(),
        op->hashProperties());

    // Remove ignorable attributes while computing hash
    SmallVector<NamedAttribute> attrs(op->getRawDictionaryAttrs().getValue());
    NamedAttribute *pos;
    while ((pos = llvm::find_if(attrs, [](NamedAttribute attr) {
      StringRef key = attr.getName().getValue();
      return key == "onnx_node_name" || key == "ResultNames";
    })) != attrs.end()) { attrs.erase(pos); }
    llvm::sort(attrs);
    hash = llvm::hash_combine(hash, ArrayRef(attrs));

    for (Value operand : op->getOperands())
      hash = llvm::hash_combine(hash, hash_value(operand));
    hash = llvm::hash_combine(hash, op->getResultTypes());
    return hash;
  }

  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    if (lhs->getName() != rhs->getName() ||
        lhs->getNumOperands() != rhs->getNumOperands() ||
        lhs->getNumResults() != rhs->getNumResults() ||
        !lhs->getName().compareOpProperties(
            lhs->getPropertiesStorage(), rhs->getPropertiesStorage()))
      return false;

    // Compare attributes, skipping ignorable attributes
    for (auto lhsA : lhs->getAttrs()) {
      StringRef key = lhsA.getName().getValue();
      if (key == "onnx_node_name" || key == "ResultNames")
        continue;
      if (!rhs->hasAttr(key) || rhs->getAttr(key) != lhsA.getValue())
        return false;
    }

    for (auto [lhsO, rhsO] :
        llvm::zip(lhs->getOperands(), rhs->getOperands())) {
      if (lhsO != rhsO)
        return false;
    }

    for (auto [lhsR, rhsR] : llvm::zip(lhs->getResults(), rhs->getResults())) {
      if (lhsR.getType() != rhsR.getType())
        return false;
    }

    return true;
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
