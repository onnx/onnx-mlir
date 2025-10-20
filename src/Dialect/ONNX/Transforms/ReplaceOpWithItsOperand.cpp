/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ReplaceOpWithItsOperand.cpp - Remove masking in Attention layer
//-----===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This pass will ignore the use of attention_mask argument/input in a function
// operation. In particular, it will replace AddOp(x, mask) by x.
//
//===----------------------------------------------------------------------===//

#include <regex>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ReplaceOpWithItsOperandPass
    : public PassWrapper<ReplaceOpWithItsOperandPass,
          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceOpWithItsOperandPass)

  ReplaceOpWithItsOperandPass() = default;
  ReplaceOpWithItsOperandPass(const ReplaceOpWithItsOperandPass &pass)
      : mlir::PassWrapper<ReplaceOpWithItsOperandPass,
            OperationPass<func::FuncOp>>() {}
  ReplaceOpWithItsOperandPass(std::vector<std::string> nodeNameRegexList) {
    this->nodeNameRegexList = nodeNameRegexList;
  };

  StringRef getArgument() const override {
    return "replace-op-with-its-operand";
  }

  StringRef getDescription() const override {
    return "Replace an operation's result by one of its operand. Only support "
           "operations that have one result.";
  }

  // Usage: onnx-mlir-opt
  // --replace-op-with-its-operand='list-id-regex=1:mask list-id-regex=0:relu'
  ListOption<std::string> nodeNameRegexList{*this, "list-id-regex",
      llvm::cl::desc(
          "A list of node name regex in the form of input_id:node_name_regex")};

  void runOnOperation() final;
};

void ReplaceOpWithItsOperandPass::runOnOperation() {
  func::FuncOp function = getOperation();

  function.walk([&](Operation *op) -> WalkResult {
    // Only deal with ONNX ops.
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return WalkResult::advance();

    // The op has only one result.
    if (op->getNumResults() != 1)
      return WalkResult::advance();

    // Check if the operation has onnx_node_name attribute.
    auto nodeNameAttr = op->getAttrOfType<mlir::StringAttr>("onnx_node_name");
    if (!nodeNameAttr)
      return WalkResult::advance();
    StringRef opNodeName = nodeNameAttr.getValue();
    if (opNodeName.empty())
      return WalkResult::advance();

    // Match node name. Return when found the first one.
    for (std::string IdNodenameRegex : this->nodeNameRegexList) {
      size_t pos = IdNodenameRegex.find(":");
      std::string inputID = IdNodenameRegex.substr(0, pos);
      std::string nodeNameRegex = IdNodenameRegex.substr(pos + 1);
      if (std::regex_match(opNodeName.str(), std::regex(nodeNameRegex))) {
        int64_t id = std::stoi(inputID);
        op->getResult(0).replaceAllUsesWith(op->getOperand(id));
        return WalkResult::advance();
      }
    }

    return WalkResult::advance();
  });
}

} // namespace onnx_mlir

std::unique_ptr<mlir::Pass> onnx_mlir::createReplaceOpWithItsOperandPass(
    std::vector<std::string> nodeNameRegexList) {
  return std::make_unique<ReplaceOpWithItsOperandPass>(nodeNameRegexList);
}
