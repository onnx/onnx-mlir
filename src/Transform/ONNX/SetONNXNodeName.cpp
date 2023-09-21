/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- SetONNXNodeName.cpp - ONNX high level transformation --------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This pass is to set onnx_node_name attribute for ONNX operations if the
// attribute is absent.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "set-onnx-node-name"

using namespace mlir;
using namespace onnx_mlir;

namespace {

struct SetONNXNodeNamePass
    : public PassWrapper<SetONNXNodeNamePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetONNXNodeNamePass)

  StringRef getArgument() const override { return "set-onnx-node-name"; }

  StringRef getDescription() const override {
    return "Set onnx_node_name attribute for ONNX operations if the attribute "
           "is absent";
  }

  void runOnOperation() final;

  std::string generateNodeName(Operation *op);

private:
  uint64_t id = 0;
  llvm::SmallSet<std::string, 32> nodeNames;
  llvm::SmallSet<Operation *, 32> opsNeedNodeName;
  std::string nodeNameAttr = "onnx_node_name";
};

void SetONNXNodeNamePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();

  // Collect
  // - all onnx_node_name strings and
  // - operations that have an empty onnx_node_name or a duplicated name.
  moduleOp.walk([&](Operation *op) -> WalkResult {
    // Only deal with ONNX ops.
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return WalkResult::advance();
    // No need onnx_node_name for these ops.
    if (isa<ONNXEntryPointOp, ONNXReturnOp, ONNXConstantOp>(op))
      return WalkResult::advance();
    StringAttr nodeName = op->getAttrOfType<mlir::StringAttr>(nodeNameAttr);
    if (nodeName && !nodeName.getValue().empty()) {
      std::string s = nodeName.getValue().str();
      bool succeeded = nodeNames.insert(s).second;
      if (!succeeded) {
        llvm::outs() << "Duplicated " << nodeNameAttr << ": " << s
                     << ". It will be updated with a new string.\n";
        opsNeedNodeName.insert(op);
      }
    } else
      opsNeedNodeName.insert(op);
    return WalkResult::advance();
  });

  // Nothing to do if all operations have onnx_node_name.
  if (opsNeedNodeName.size() == 0)
    return;

  // Set onnx_node_name for each operation.
  for (Operation *op : opsNeedNodeName) {
    std::string s = generateNodeName(op);
    op->setAttr(nodeNameAttr, StringAttr::get(context, s));
  }
}

std::string SetONNXNodeNamePass::generateNodeName(Operation *op) {
  std::string opName = op->getName().getStringRef().str();
  // Try to use the common way of setting node name that is used in the
  // instrumentation. Not use fileLineLoc since it includes a fixed model file
  // name and contains symbols that are not friendly to users.
  std::string res = getNodeNameInPresenceOfOpt(op, /*useFileLine=*/false);
  if (res == "NOTSET")
    res = "";

  // Try to relate this op to the op that produces its inputs.
  // In that case, onnx_node_name = first_input_onnx_node_name + opName + id.
  if (res.empty() && op->getOperands().size() > 0) {
    // Use the first input if it has onnx_node_name.
    if (auto firstInputOp = dyn_cast_if_present<Operation *>(
            op->getOperands()[0].getDefiningOp())) {
      if (!isa<ONNXConstantOp, ONNXDimOp>(firstInputOp) &&
          !opsNeedNodeName.contains(firstInputOp)) {
        StringAttr nodeName =
            firstInputOp->getAttrOfType<mlir::StringAttr>(nodeNameAttr);
        if (nodeName && !nodeName.getValue().empty()) {
          res = nodeName.getValue().str() + "_" + opName;
          // Name is too long, give up this way.
          if (res.length() > 256)
            res = "";
        }
      }
    }
  }
  // Otherwise, onnx_node_name = opName + id.
  if (res.empty())
    res = opName;

  // Append ID to make it unique.
  res += "_" + std::to_string(id++);

  // The new name may exist, try again to get a unique name.
  while (nodeNames.contains(res))
    res = opName + "_" + std::to_string(id++);
  return res;
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a SetONNXNodeName pass.
 */
std::unique_ptr<mlir::Pass> createSetONNXNodeNamePass() {
  return std::make_unique<SetONNXNodeNamePass>();
}

} // namespace onnx_mlir
