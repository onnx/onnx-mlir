/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- GenerateConfigFile.cpp - Generate Config File for NNPA ------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This pass generates a JSON configuration file by reading the current IR.
// The configuration file can be used for device placement, quantization, or
// other NNPA-specific settings.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#include "src/Accelerators/NNPA/Compiler/NNPAJsonConfigObject.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "generate-config-file"

using namespace mlir;
using namespace onnx_mlir;

namespace {

struct GenerateConfigFilePass
    : public PassWrapper<GenerateConfigFilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateConfigFilePass)

  GenerateConfigFilePass() = default;
  GenerateConfigFilePass(const GenerateConfigFilePass &pass)
      : PassWrapper<GenerateConfigFilePass, OperationPass<ModuleOp>>() {}
  GenerateConfigFilePass(std::string outputConfigFile) {
    this->outputConfigFile = outputConfigFile;
  }

  StringRef getArgument() const override { return "generate-config-file"; }

  StringRef getDescription() const override {
    return "Generate a JSON configuration file from the current IR";
  }

  Option<std::string> outputConfigFile{*this, "output-config-file",
      llvm::cl::desc(
          "Path to save the generated configuration file in JSON format"),
      llvm::cl::init("")};

  void runOnOperation() final;

private:
  ModuleOp module;
  MLIRContext *context = nullptr;
  // Keep target operations to avoid walking through the module again.
  // Use vector to keep the order deterministic.
  SmallVector<Operation *, 32> ops;

  // JSON keys.
  std::string DEVICE_PLACEMENT_KEY = "device_placement";
  std::string QUANTIZATION_KEY = "quantization";

  // Exclude these operations from config generation.
  bool isExcludedOp(Operation *op) {
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return true;
    // No annotation for these ops.
    if (isa<ONNXEntryPointOp, ONNXReturnOp, ONNXConstantOp>(op))
      return true;
    return false;
  }
};

void GenerateConfigFilePass::runOnOperation() {
  this->module = getOperation();
  this->context = &getContext();

  // Collect target operations from the module.
  module.walk([&](Operation *op) {
    if (!isExcludedOp(op))
      ops.emplace_back(op);
  });

  if (!outputConfigFile.empty()) {
    NNPAJsonConfigObject configObj;
    
    configObj.writeOpsConfig(ops,
        [&](mlir::Operation *op, llvm::json::Object &match,
            llvm::json::Object &rewrite) -> bool {
          bool hasConfig = false;

          // Add device to rewrite if present.
          if (auto deviceAttr = op->getAttrOfType<mlir::StringAttr>(
                  NNPAJsonConfigObject::DEVICE_ATTR)) {
            std::string deviceStr = deviceAttr.getValue().str();
            if (!deviceStr.empty()) {
              rewrite[NNPAJsonConfigObject::DEVICE_KEY] = deviceStr;
              hasConfig = true;
            }
          }

          // Add quantize to rewrite if present.
          if (auto quantAttr = op->getAttrOfType<mlir::BoolAttr>(
                  NNPAJsonConfigObject::QUANTIZE_ATTR)) {
            rewrite[NNPAJsonConfigObject::QUANTIZE_KEY] = quantAttr.getValue();
            hasConfig = true;
          }

          return hasConfig;
        });
    
    // Store the configuration to file.
    configObj.storeToFile(outputConfigFile);
  }
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a GenerateConfigFile pass.
 */
std::unique_ptr<mlir::Pass> createGenerateConfigFilePass() {
  return std::make_unique<GenerateConfigFilePass>();
}

std::unique_ptr<mlir::Pass> createGenerateConfigFilePass(
    std::string outputConfigFile) {
  return std::make_unique<GenerateConfigFilePass>(outputConfigFile);
}

} // namespace onnx_mlir
