/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SetDiscardableAttr.cpp - Set Discardable Attributes --------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass that sets discardable attributes on operations
// based on the provided attribute names.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_SETDISCARDABLEATTR
#include "src/Transform/Passes.h.inc"

/*!
 * This pass sets discardable attributes on operations
 */

class SetDiscardableAttr : public impl::SetDiscardableAttrBase<SetDiscardableAttr> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetDiscardableAttr)

  SetDiscardableAttr() = default;

  SetDiscardableAttr(const SetDiscardableAttr &pass)
      : impl::SetDiscardableAttrBase<SetDiscardableAttr>() {}

  SetDiscardableAttr(const std::vector<std::string> &names) {
    this->attrNames = names;
  }

  StringRef getArgument() const override { return "set-discardable-attr"; }

  StringRef getDescription() const override {
    return "Set discardable attributes on operations";
  }

  void runOnOperation() override {
    Operation *rootOp = getOperation();
    MLIRContext *context = rootOp->getContext();

    // Early return if no attribute names provided
    if (attrNames.empty()) {
      return;
    }

    // Walk through all operations in the module/function
    rootOp->walk([&](Operation *op) {
      // Process each attribute name from the option
      for (const std::string &attrName : attrNames) {
        // Check if the operation has this attribute
        if (op->hasAttr(attrName)) {
          // Get the existing attribute
          Attribute attr = op->getAttr(attrName);

          // Set it as a discardable attribute (prefixed with "_.")
          // Remove the original non-discardable version
          op->removeAttr(attrName);

          // Add as discardable (with "_." prefix)
          std::string discardableName = "_." + attrName;
          op->setAttr(discardableName, attr);
        }
      }
    });
  }
};

} // namespace onnx_mlir

/*!
 * Create a SetDiscardableAttr pass.
 */
namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createSetDiscardableAttrPass() {
  return std::make_unique<SetDiscardableAttr>();
}

std::unique_ptr<mlir::Pass> createSetDiscardableAttrPass(
    const std::vector<std::string> &attrNames) {
  return std::make_unique<SetDiscardableAttr>(attrNames);
}

} // namespace onnx_mlir
