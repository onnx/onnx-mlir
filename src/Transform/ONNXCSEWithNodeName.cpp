/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ONNXCSEWithNodeName.cpp - ONNX CSE with Node Names ----------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass that performs Common Subexpression Elimination
// (CSE) on ONNX operations while preserving onnx_node_name attributes.
//
// The pass works in three phases:
// 1. Save onnx_node_name attributes and remove them from operations
// 2. Run MLIR's built-in CSE pass
// 3. Restore node names (keeping first occurrence's name for merged ops)
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "onnx-cse-with-node-name"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_ONNXCSEWITHNODENAMEPASS
#include "src/Transform/Passes.h.inc"

} // namespace onnx_mlir

namespace {

struct ONNXCSEWithNodeNamePass
    : public onnx_mlir::impl::ONNXCSEWithNodeNamePassBase<
          ONNXCSEWithNodeNamePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXCSEWithNodeNamePass)

  void runOnOperation() override;

private:
  // Map from operation to its original onnx_node_name.
  llvm::DenseMap<Operation *, StringAttr> nodeNameMap;

  // Attribute name for node names.
  static constexpr const char *nodeNameAttr = "onnx_node_name";

  // Save onnx_node_name attributes and remove them from operations.
  void saveAndRemoveNodeNames(ModuleOp module);

  // Run MLIR CSE pass.
  LogicalResult runCSE(ModuleOp module);

  // Restore node names to surviving operations.
  void restoreNodeNames(ModuleOp module);

  // Check if operation is an ONNX operation that needs node name.
  bool needsNodeName(Operation *op);
};

bool ONNXCSEWithNodeNamePass::needsNodeName(Operation *op) {
  // Only process ONNX dialect operations.
  if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
    return false;

  // Skip operations that don't need node names.
  if (isa<ONNXEntryPointOp, ONNXReturnOp, ONNXConstantOp>(op))
    return false;

  return true;
}

void ONNXCSEWithNodeNamePass::saveAndRemoveNodeNames(ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Saving and removing onnx_node_name attributes\n");

  module.walk([&](Operation *op) {
    if (!needsNodeName(op))
      return WalkResult::advance();

    // Save onnx_node_name if present.
    if (auto nodeName = op->getAttrOfType<StringAttr>(nodeNameAttr)) {
      LLVM_DEBUG(llvm::dbgs() << "  Saving node name: " << nodeName.getValue()
                              << " for op: " << op->getName() << "\n");
      nodeNameMap[op] = nodeName;
      op->removeAttr(nodeNameAttr);
    }
    return WalkResult::advance();
  });

  LLVM_DEBUG(llvm::dbgs() << "Saved " << nodeNameMap.size() << " node names\n");
}

LogicalResult ONNXCSEWithNodeNamePass::runCSE(ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Running CSE pass\n");

  // Create and run CSE pass.
  PassManager pm(module.getContext());
  pm.addPass(mlir::createCSEPass());

  if (failed(pm.run(module))) {
    LLVM_DEBUG(llvm::dbgs() << "CSE pass failed\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "CSE pass completed successfully\n");
  return success();
}

void ONNXCSEWithNodeNamePass::restoreNodeNames(ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Restoring onnx_node_name attributes\n");

  int restoredCount = 0;
  module.walk([&](Operation *op) {
    if (!needsNodeName(op))
      return WalkResult::advance();

    // Restore name if this operation survived CSE.
    auto it = nodeNameMap.find(op);
    if (it != nodeNameMap.end()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Restoring node name: " << it->second.getValue()
                 << " for op: " << op->getName() << "\n");
      op->setAttr(nodeNameAttr, it->second);
      restoredCount++;
    }
    return WalkResult::advance();
  });

  LLVM_DEBUG(llvm::dbgs() << "Restored " << restoredCount
                          << " node names out of " << nodeNameMap.size()
                          << " saved\n");

  // Log eliminated operations.
  int eliminatedCount = nodeNameMap.size() - restoredCount;
  if (eliminatedCount > 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "CSE eliminated " << eliminatedCount << " operations\n");
  }
}

void ONNXCSEWithNodeNamePass::runOnOperation() {
  ModuleOp module = getOperation();

  // Phase 1: Save and remove node names.
  saveAndRemoveNodeNames(module);

  // Phase 2: Run CSE.
  if (failed(runCSE(module))) {
    signalPassFailure();
    return;
  }

  // Phase 3: Restore node names.
  restoreNodeNames(module);

  // Clear the map to free memory.
  nodeNameMap.clear();
}

} // namespace
