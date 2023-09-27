/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- DevicePlacement.cpp - Device Placement for NNPA -------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This pass is to set device (CPU, or NNPA) for each operation in ONNX level.
// Device placement can be decided by:
// - user configuration file if given
// - a cost model
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

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "device-placement"

using namespace mlir;
using namespace onnx_mlir;

namespace {

struct DevicePlacementPass
    : public PassWrapper<DevicePlacementPass, OperationPass<ModuleOp>> {
  using OpSetType = DenseSet<Operation *>;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DevicePlacementPass)

  StringRef getArgument() const override { return "device-placement"; }

  StringRef getDescription() const override {
    return "Device placement for NNPA";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  DevicePlacementPass() = default;
  DevicePlacementPass(const DevicePlacementPass &pass)
      : PassWrapper<DevicePlacementPass, OperationPass<ModuleOp>>() {}
  DevicePlacementPass(std::string loadConfigFile, std::string saveConfigFile) {
    this->loadConfigFile = loadConfigFile;
    this->saveConfigFile = saveConfigFile;
  }

  Option<std::string> saveConfigFile{*this, "save-config-file",
      llvm::cl::desc("Path to save a device configuration file in JSON format"),
      llvm::cl::init("")};

  Option<std::string> loadConfigFile{*this, "load-config-file",
      llvm::cl::desc("Path to load a device configuration file in JSON format"),
      llvm::cl::init("")};

  bool isExcludedOp(Operation *op) {
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return true;
    // No annotation for these ops.
    if (isa<ONNXEntryPointOp, ONNXReturnOp, ONNXConstantOp>(op))
      return true;
    return false;
  }

  void runOnOperation() final;
  void loadConfigFromJSONFile(ModuleOp &module);
  void saveConfigToJSONFile(ModuleOp &module);

private:
  std::string DEVICE_KEY = "device";
  std::string DEVICE_PLACEMENT_KEY = "device_placement";
  std::string NODE_TYPE_KEY = "node_type";
  std::string ONNX_NODE_NAME_KEY = "onnx_node_name";
};

void DevicePlacementPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  // Run the unknown dimension analysis to help check equality of unknown
  // dimensions at compile time.
  DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

  // Cost model and user configuration file go here if it's given.
  // (Reserved for cost model and user configuration file)
  if (!loadConfigFile.empty())
    loadConfigFromJSONFile(module);

  // Run patterns that converts ONNX to ZHigh with analysis mode to collect
  // operations that are not converted. Those non-converted ops are running on
  // the host instead of accelerator.
  // Keep the order of calling pass synced with RewriteONNXForZHigh.cpp and
  // ONNXToZHigh.cpp.

  OpSetType legalizedOps1, legalizedOps2, legalizedOps3;

  ConversionTarget target(*context);
  target.addLegalDialect<ONNXDialect, func::FuncDialect, arith::ArithDialect>();

  // Call RewriteONNXForZHigh pass.
  RewritePatternSet Patterns1(context);
  getRewriteONNXForZHighPatterns(Patterns1, &dimAnalysis);
  getRewriteONNXForZHighDynamicallyLegal(&target, &dimAnalysis);
  (void)applyAnalysisConversion(
      module, target, std::move(Patterns1), legalizedOps1);

  // Call ONNXToZHigh pass for lowering multiple ONNX ops at once to ZHigh.
  // E.g. `onnx.ReLu (onnx.Conv)` to zhigh.Conv.
  RewritePatternSet Patterns2(context);
  getONNXToZHighOneOpPatterns(Patterns2);
  (void)applyAnalysisConversion(
      module, target, std::move(Patterns2), legalizedOps2);

  // Call ONNXToZHigh pass for lowering a single ONNX op to ZHigh.
  RewritePatternSet Patterns3(context);
  getONNXToZHighOneOpPatterns(Patterns3);
  getONNXToZHighOneOpDynamicallyLegal(&target, &dimAnalysis);
  (void)applyAnalysisConversion(
      module, target, std::move(Patterns3), legalizedOps3);

  // Get the legalized ops that will run on the host.
  OpSetType cpuOps = llvm::set_intersection(
      legalizedOps1, llvm::set_intersection(legalizedOps2, legalizedOps3));

  // Now annotate accelerator operations in the IR with `device` attribute,
  // according to the compiler decision.
  module.walk([&](Operation *op) -> WalkResult {
    if (isExcludedOp(op))
      return WalkResult::advance();
    // Set device if it is empty or unavailable.
    StringAttr device = op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
    if (!device || device.getValue().empty()) {
      if (!cpuOps.contains(op))
        op->setAttr(DEVICE_ATTRIBUTE, StringAttr::get(context, NNPA_DEVICE));
    }
    return WalkResult::advance();
  });

  // Create a JSON configuration file if required.
  if (!saveConfigFile.empty())
    saveConfigToJSONFile(module);
}

// Global object to ease error reporting, it consumes errors and crash the
// application with a meaningful message.
static llvm::ExitOnError ExitOnErr;

void DevicePlacementPass::loadConfigFromJSONFile(ModuleOp &module) {
  auto Buf = ExitOnErr(errorOrToExpected(
      llvm::MemoryBuffer::getFile(loadConfigFile, /*bool IsText=*/true,
          /*RequiresNullTerminator=*/false)));
  auto jsonFile = ExitOnErr(llvm::json::parse(Buf->getBuffer()));
  llvm::json::Object *jsonContent = jsonFile.getAsObject();
  llvm::json::Array *jsonArr = jsonContent->getArray(DEVICE_PLACEMENT_KEY);
  if (!jsonArr || jsonArr->empty())
    return;

  // Collect operations to work on.
  OpSetType workingOps;
  module.walk([&](Operation *op) {
    if (!isExcludedOp(op))
      workingOps.insert(op);
  });

  // To reduce complexity, once an operation is assigned a device, we remove it
  // from the set workingOps.
  for (llvm::json::Value v : *jsonArr) {
    llvm::json::Object *vobj = v.getAsObject();
    StringRef device = vobj->getString(DEVICE_KEY).value();
    StringRef nodeType = vobj->getString(NODE_TYPE_KEY).value();
    StringRef nodeName = vobj->getString(ONNX_NODE_NAME_KEY).value();
    LLVM_DEBUG(llvm::dbgs()
               << "device: " << device.str() << ", nodeType: " << nodeType.str()
               << ", nodeName: " << nodeName.str() << "\n");
    OpSetType updatedOps;
    for (Operation *op : workingOps) {
      StringRef opNodeType = op->getName().getStringRef();
      StringRef opNodeName =
          op->getAttrOfType<mlir::StringAttr>("onnx_node_name").getValue();

      if (!std::regex_match(opNodeType.str(), std::regex(nodeType.str())))
        continue;
      if (!std::regex_match(opNodeName.str(), std::regex(nodeName.str())))
        continue;

      op->setAttr(
          DEVICE_ATTRIBUTE, StringAttr::get(module.getContext(), device));
      updatedOps.insert(op);

      // if (opNodeType.equals_insensitive(nodeType.value()) &&
      //     opNodeName.equals_insensitive(nodeName.value())) {
      //   op->setAttr(DEVICE_ATTRIBUTE,
      //       StringAttr::get(module.getContext(), device.value()));
      //   updatedOps.insert(op);
      // }
    }
    workingOps = llvm::set_difference(workingOps, updatedOps);
  }
}

void DevicePlacementPass::saveConfigToJSONFile(ModuleOp &module) {
  llvm::json::Array jsonArr;
  module.walk([&](Operation *op) -> WalkResult {
    if (isExcludedOp(op))
      return WalkResult::advance();
    // Create a JSON object for this operation.
    std::string deviceStr =
        op->getAttrOfType<mlir::StringAttr>("device")
            ? op->getAttrOfType<mlir::StringAttr>("device").getValue().str()
            : "";
    std::string nodeTypeStr = op->getName().getStringRef().str();
    std::string nodeNameStr =
        op->getAttrOfType<mlir::StringAttr>("onnx_node_name")
            ? op->getAttrOfType<mlir::StringAttr>("onnx_node_name")
                  .getValue()
                  .str()
            : "";
    llvm::json::Value jsonObj = llvm::json::Object{
        {DEVICE_KEY, deviceStr},
        {NODE_TYPE_KEY, nodeTypeStr},
        {ONNX_NODE_NAME_KEY, nodeNameStr},
    };
    jsonArr.emplace_back(jsonObj);

    return WalkResult::advance();
  });

  // Create a JSON configuration file.
  if (!saveConfigFile.empty()) {
    llvm::json::Object jsonContent{
        {DEVICE_PLACEMENT_KEY, llvm::json::Value(std::move(jsonArr))}};
    std::error_code EC;
    llvm::raw_fd_ostream jsonOS(saveConfigFile, EC);
    if (EC)
      report_fatal_error("Error saving device placement json file : " +
                         StringRef(EC.message()));
    jsonOS << llvm::json::Value(std::move(jsonContent)) << "\n";
    jsonOS.close();
  }
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a DevicePlacement pass.
 */
std::unique_ptr<mlir::Pass> createDevicePlacementPass() {
  return std::make_unique<DevicePlacementPass>();
}

std::unique_ptr<mlir::Pass> createDevicePlacementPass(
    std::string loadConfigFile, std::string saveConfigFile) {
  return std::make_unique<DevicePlacementPass>(loadConfigFile, saveConfigFile);
}

} // namespace onnx_mlir
