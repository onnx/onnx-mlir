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

#include <regex>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "quant-op-selection"

using namespace mlir;
using namespace onnx_mlir;

namespace {

// Global object to ease error reporting, it consumes errors and crash the
// application with a meaningful message.
static llvm::ExitOnError ExitOnErr;

struct QuantOpSelectionPass
    : public PassWrapper<QuantOpSelectionPass, OperationPass<ModuleOp>> {
  using OpSetType = DenseSet<Operation *>;

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
  std::string QUANTIZATION_KEY = "quantization_ops";
  std::string NODE_TYPE_KEY = "node_type";
  std::string ONNX_NODE_NAME_KEY = "onnx_node_name";

  // Exclude these operations from quantization.
  bool isExcludedOp(Operation *op) {
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return true;
    // No annotation for these ops.
    if (isa<ONNXEntryPointOp, ONNXReturnOp, ONNXConstantOp>(op))
      return true;
    return false;
  }

  // Functions to load/save quantization settings from/to a JSON file.
  // JSON file example:
  // ```json
  // {
  //   "quantization_ops": [
  //     {
  //       "node_type": "onnx.Relu",
  //       "onnx_node_name": "Relu_[1,2]"
  //     },
  //     {
  //       "node_type": "onnx.Sigmoid",
  //       "onnx_node_name": ".*"
  //     }
  //   ]
  // }
  // ```
  void loadConfigFromJSONFile();
  void saveConfigToJSONFile();
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
  if (!loadConfigFile.empty())
    loadConfigFromJSONFile();

  // TODO: Before saving the configuration, we need to know all ops that are
  // quantized by the compiler (for example, there is no quantization info in
  // the loading config file, but the compiler decides to quantize some ops).
  // How to obtain such info from the compiler's decision?
  //
  // Currently, only quantization info for ops in the loading config file are
  // saved.

  // Create a JSON configuration file if required.
  if (!saveConfigFile.empty())
    saveConfigToJSONFile();
}

void QuantOpSelectionPass::loadConfigFromJSONFile() {
  auto Buf = ExitOnErr(errorOrToExpected(
      llvm::MemoryBuffer::getFile(loadConfigFile, /*bool IsText=*/true,
          /*RequiresNullTerminator=*/false)));
  auto jsonFile = ExitOnErr(llvm::json::parse(Buf->getBuffer()));
  llvm::json::Object *jsonContent = jsonFile.getAsObject();
  llvm::json::Array *jsonArr = jsonContent->getArray(QUANTIZATION_KEY);
  if (!jsonArr || jsonArr->empty())
    return;

  // Collect operations to work on.
  OpSetType workingOps(ops.begin(), ops.end());
  // Go over operations in the JSON and find matched operation in the IR.
  for (llvm::json::Value v : *jsonArr) {
    llvm::json::Object *vobj = v.getAsObject();
    StringRef nodeType = vobj->getString(NODE_TYPE_KEY).value();
    std::optional<StringRef> nodeName = vobj->getString(ONNX_NODE_NAME_KEY);
    OpSetType updatedOps;
    for (Operation *op : workingOps) {
      StringRef opNodeType = op->getName().getStringRef();
      StringRef opNodeName =
          op->getAttrOfType<mlir::StringAttr>("onnx_node_name").getValue();
      // Match operation.
      if (!std::regex_match(opNodeType.str(), std::regex(nodeType.str())))
        continue;
      if (nodeName.has_value() && !std::regex_match(opNodeName.str(),
                                      std::regex(nodeName.value().str())))
        continue;
      // Set quantization.
      op->setAttr(QUANT_ATTRIBUTE, BoolAttr::get(module.getContext(), true));
      updatedOps.insert(op);
    }
    // To reduce complexity, once an operation is assigned the quantize
    // attribute, we remove it from the set workingOps.
    workingOps = llvm::set_difference(workingOps, updatedOps);
  }
}

void QuantOpSelectionPass::saveConfigToJSONFile() {
  // Parsing the module to JSON object.
  llvm::json::Array jsonArr;
  for (Operation *op : ops) {
    BoolAttr attr = op->getAttrOfType<mlir::BoolAttr>(QUANT_ATTRIBUTE);
    bool shouldQuant = attr ? attr.getValue() : false;
    if (!shouldQuant)
      continue;
    // Create a JSON object for this operation.
    std::string nodeTypeStr = op->getName().getStringRef().str();
    std::string nodeNameStr =
        op->getAttrOfType<mlir::StringAttr>("onnx_node_name")
            ? op->getAttrOfType<mlir::StringAttr>("onnx_node_name")
                  .getValue()
                  .str()
            : "";
    llvm::json::Value jsonObj = llvm::json::Object{
        {NODE_TYPE_KEY, nodeTypeStr},
        {ONNX_NODE_NAME_KEY, nodeNameStr},
    };
    jsonArr.emplace_back(jsonObj);
  }
  addOrAppendJSonObjectToFile(
      QUANTIZATION_KEY, llvm::json::Value(std::move(jsonArr)), saveConfigFile);
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
