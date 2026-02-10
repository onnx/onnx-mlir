/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ConvertAttrToDiscardable.cpp - Convert Attributes to Discardable -===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass that converts attributes to discardable form
// by prefixing them with '_.', based on the provided attribute names.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_CONVERTATTRTODISCARDABLE
#include "src/Transform/Passes.h.inc"

/*!
 * This pass converts attributes to discardable form
 */

class ConvertAttrToDiscardable : public impl::ConvertAttrToDiscardableBase<ConvertAttrToDiscardable> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAttrToDiscardable)

  ConvertAttrToDiscardable() = default;

  ConvertAttrToDiscardable(const ConvertAttrToDiscardable &pass)
      : impl::ConvertAttrToDiscardableBase<ConvertAttrToDiscardable>() {}

  ConvertAttrToDiscardable(const std::vector<std::string> &names) {
    this->attrNames = names;
  }

  StringRef getArgument() const override { return "convert-attr-to-discardable"; }

  StringRef getDescription() const override {
    return "Convert attributes to discardable form";
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

          // Remove the original non-discardable version
          op->removeAttr(attrName);

          // Set as a discardable attribute (automatically prefixed with "_.")
          op->setDiscardableAttr(attrName, attr);
        }
      }
    });
  }
};

} // namespace onnx_mlir

/*!
 * Create a ConvertAttrToDiscardable pass.
 */
namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createConvertAttrToDiscardablePass() {
  return std::make_unique<ConvertAttrToDiscardable>();
}

std::unique_ptr<mlir::Pass> createConvertAttrToDiscardablePass(
    const std::vector<std::string> &attrNames) {
  return std::make_unique<ConvertAttrToDiscardable>(attrNames);
}

} // namespace onnx_mlir
