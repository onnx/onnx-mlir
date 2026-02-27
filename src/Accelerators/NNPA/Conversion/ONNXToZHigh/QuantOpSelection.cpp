/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- QuantOpSelection.cpp - Select Ops for Quantization ----------===//
//
// Copyright 2025-2026 The IBM Research Authors.
//
// =============================================================================
//
// This pass is to add a boolean attribute, namely `quantize`, to onnx
// operations based on a json configuration file.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#include "src/Accelerators/NNPA/Compiler/NNPAJsonConfigObject.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "quant-op-selection"

using namespace mlir;
using namespace onnx_mlir;

namespace {

struct QuantOpSelectionPass
    : public PassWrapper<QuantOpSelectionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantOpSelectionPass)

  QuantOpSelectionPass() = default;
  QuantOpSelectionPass(const QuantOpSelectionPass &pass)
      : PassWrapper<QuantOpSelectionPass, OperationPass<ModuleOp>>() {}
  QuantOpSelectionPass(std::string loadConfigFile, std::string saveConfigFile) {
    this->loadConfigFile = loadConfigFile;
    this->saveConfigFile = saveConfigFile;
  }

  StringRef getArgument() const override { return "nnpa-quant-ops-selection"; }

  StringRef getDescription() const override {
    return "Select ops for quantization";
  }

  Option<std::string> saveConfigFile{*this, "save-config-file",
      llvm::cl::desc(
          "Path to save a quantization configuration file in JSON format"),
      llvm::cl::init("")};

  Option<std::string> loadConfigFile{*this, "load-config-file",
      llvm::cl::desc(
          "Path to load a quantization configuration file in JSON format"),
      llvm::cl::init("")};

  void runOnOperation() final;

private:
  ModuleOp module;
  MLIRContext *context = nullptr;
  // Keep target operations to avoid walking through the module again.
  // Use vector to keep the order deterministic.
  SmallVector<Operation *, 32> ops;

  // JSON configuration object - either points to global or local instance.
  NNPAJsonConfigObject *configObject;
  // Local config object storage (only used when loadConfigFile is provided).
  std::unique_ptr<NNPAJsonConfigObject> localConfigObject;

  // Exclude these operations from quantization.
  bool isExcludedOp(Operation *op) {
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return true;
    // No annotation for these ops.
    if (isa<ONNXEntryPointOp, ONNXReturnOp, ONNXConstantOp>(op))
      return true;
    return false;
  }
};

void QuantOpSelectionPass::runOnOperation() {
  this->module = getOperation();
  this->context = &getContext();

  // Collects target operations from the module.
  module.walk([&](Operation *op) {
    if (!isExcludedOp(op))
      ops.emplace_back(op);
  });

  // Initialize configObject pointer based on loadConfigFile.
  // Note: This must be done here, not in constructor, because loadConfigFile
  // is an Option that gets initialized by MLIR after constructor runs.
  if (!loadConfigFile.empty()) {
    // Create a local config object and load from the specified file.
    localConfigObject = std::make_unique<NNPAJsonConfigObject>();
    if (!localConfigObject->loadFromFile(loadConfigFile)) {
      llvm::errs() << "Warning: Failed to load config file: " << loadConfigFile
                   << "\n";
    }
    configObject = localConfigObject.get();
  } else {
    // Use the global configuration object.
    configObject = &getGlobalNNPAConfig();
  }

  // Cost model and user configuration file go here if it's given.
  // Use the configObject pointer which points to either local or global config.
  if (configObject && !configObject->empty()) {
    // Apply configuration to ONNX ops.
    configObject->applyConfigToOps(
        ops, [&](llvm::json::Object *rewriteObj, mlir::Operation *op) {
          if (auto quantize =
                  rewriteObj->getBoolean(NNPAJsonConfigObject::QUANTIZE_KEY)) {
            op->setAttr(NNPAJsonConfigObject::QUANTIZE_ATTR,
                BoolAttr::get(module.getContext(), *quantize));
          }
        });
  }

  // TODO: Before saving the configuration, we need to know all ops that are
  // quantized by the compiler (for example, there is no quantization info in
  // the loading config file, but the compiler decides to quantize some ops).
  // How to obtain such info from the compiler's decision?
  //
  // Currently, only quantization info for ops in the loading config file are
  // saved.

  // Create a JSON configuration file if required.
  if (!saveConfigFile.empty()) {
    configObject->writeOpsConfig(
        ops, [&](mlir::Operation *op, llvm::json::Object &rewrite) -> bool {
          BoolAttr quantAttr = op->getAttrOfType<mlir::BoolAttr>(
              NNPAJsonConfigObject::QUANTIZE_ATTR);
          if (!quantAttr)
            return false;
          // Add quantize to rewrite.
          rewrite[NNPAJsonConfigObject::QUANTIZE_KEY] = quantAttr.getValue();
          return true;
        });

    // Store the configuration to file.
    configObject->storeToFile(saveConfigFile);
  }
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a QuantOpSelection pass.
 */
std::unique_ptr<mlir::Pass> createQuantOpSelectionPass() {
  return std::make_unique<QuantOpSelectionPass>();
}

std::unique_ptr<mlir::Pass> createQuantOpSelectionPass(
    std::string loadConfigFile, std::string saveConfigFile) {
  return std::make_unique<QuantOpSelectionPass>(loadConfigFile, saveConfigFile);
}

} // namespace onnx_mlir
