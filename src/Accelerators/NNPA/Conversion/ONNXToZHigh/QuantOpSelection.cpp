/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- QuantOpSelection.cpp - Select Ops for Quantization ----------===//
//
// Copyright 2025 The IBM Research Authors.
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

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/JsonConfigFile.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
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

  // JSON keys.
  std::string QUANTIZATION_KEY = "quantization";

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

  // Cost model and user configuration file go here if it's given.
  // (Reserved for cost model and user configuration file)
  NNPAJsonConfig cfg(QUANTIZATION_KEY);
  if (!loadConfigFile.empty()) {
    // Match and update operations using the json object of key QUANTIZATION_KEY
    // in the json file by setting attribute QUANT_ATTRIBUTE for the operations.
    // The value of QUANT_ATTRIBUTE is from the json file.
    cfg.loadConfigFromFile(ops, loadConfigFile,
        [&](llvm::json::Object *jsonObj, mlir::Operation *op) {
          bool quantize = jsonObj->getBoolean(QUANT_ATTRIBUTE).value();
          op->setAttr(
              QUANT_ATTRIBUTE, BoolAttr::get(module.getContext(), quantize));
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
    // Save quantization information to a json file by adding to the existing
    // json file an json object of key QUANTIZATION_KEY.
    // Each value in the object is added a pair (QUANT_ATTRIBUTE, value) that
    // denotes the value of QUANT_ATTRIBUTE in the operation.
    cfg.saveConfigToFile(
        ops, saveConfigFile, [&](llvm::json::Object *jsonObj, Operation *op) {
          BoolAttr attr = op->getAttrOfType<mlir::BoolAttr>(QUANT_ATTRIBUTE);
          if (attr)
            jsonObj->insert({QUANT_ATTRIBUTE, attr.getValue()});
        });
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
