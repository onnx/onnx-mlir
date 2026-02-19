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
// Device placement is done via setting `device` attribute for each operation.
// Values for `device` attribute is one of the following strings:
// - "": an empty string means the compiler decides whether the operation is on
// CPU or NNPA.
// - "nnpa": the operation may run on NNPA or CPU, and the final decision is
// made by the compiler. If `device=nnpa` is the result of this device-placement
// pass, then it means the compiler thinks it is suitable for NNPA.
// - "cpu": the operation is guaranteed to run on CPU.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#include "src/Accelerators/NNPA/Compiler/NNPACompilerUtils.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/DevicePlacementHeuristic.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/JsonConfigFile.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/JsonConfigObject.hpp"
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

  DevicePlacementPass() : configObject(nullptr) {}
  DevicePlacementPass(const DevicePlacementPass &pass)
      : PassWrapper<DevicePlacementPass, OperationPass<ModuleOp>>(),
        configObject(nullptr) {
    this->placementHeuristic = QualifyingOps;
  }
  DevicePlacementPass(std::string loadConfigFile, std::string saveConfigFile,
      NNPAPlacementHeuristic placementHeuristic)
      : configObject(nullptr) {
    this->loadConfigFile = loadConfigFile;
    this->saveConfigFile = saveConfigFile;
    this->placementHeuristic = placementHeuristic;
  }

  StringRef getArgument() const override { return "device-placement"; }

  StringRef getDescription() const override {
    return "Device placement for NNPA";
  }

  Option<std::string> saveConfigFile{*this, "save-config-file",
      llvm::cl::desc("Path to save a device configuration file in JSON format"),
      llvm::cl::init("")};

  Option<std::string> loadConfigFile{*this, "load-config-file",
      llvm::cl::desc("Path to load a device configuration file in JSON format"),
      llvm::cl::init("")};

  // Placement heuristic switches (policy driven by placementHeuristic).
  NNPAPlacementHeuristic placementHeuristic;
  // Option useXXX listed in decreasing order of priority, if multiple are
  // selected.
  Option<bool> useMuchFasterWithStickOps{*this, "use-much-faster-wsu",
      llvm::cl::desc("Enable FasterOpsWithStickUnstick NNPAPlacementHeuristic"),
      llvm::cl::init(false)};
  Option<bool> useFasterWithStickOps{*this, "use-faster-wsu",
      llvm::cl::desc("Enable FasterOpsWithStickUnstick NNPAPlacementHeuristic"),
      llvm::cl::init(false)};
  Option<bool> useFasterOps{*this, "use-faster",
      llvm::cl::desc("Enable FasterOps NNPAPlacementHeuristic"),
      llvm::cl::init(false)};
  // Method to override placement using useXXX flags
  void initPlacementHeuristic() {
    if (useMuchFasterWithStickOps)
      placementHeuristic = MuchFasterOpsWSU;
    else if (useFasterWithStickOps)
      placementHeuristic = FasterOpsWSU;
    else if (useFasterOps)
      placementHeuristic = FasterOps;
  }

  void runOnOperation() final;

private:
  ModuleOp module;
  MLIRContext *context = nullptr;
  // Keep target operations to avoid walking through the module again.
  // Use vector to keep the order deterministic.
  SmallVector<Operation *, 32> ops;

  // JSON configuration object - either points to global or local instance.
  JsonConfigObject *configObject;
  // Local config object storage (only used when loadConfigFile is provided).
  std::unique_ptr<JsonConfigObject> localConfigObject;

  // JSON keys.
  std::string DEVICE_PLACEMENT_KEY = "device_placement";

  // Exclude these operations from device placement.
  bool isExcludedOp(Operation *op) {
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return true;
    // No annotation for these ops.
    if (isa<ONNXEntryPointOp, ONNXReturnOp, ONNXConstantOp>(op))
      return true;
    return false;
  }
};

void DevicePlacementPass::runOnOperation() {
  this->module = getOperation();
  this->context = &getContext();

  // Disable reporting on NNPA unsupported ops in this pass even if
  // `-opt-report=NNPAUnsupportedOps` is specified..
  ONNXToZHighLoweringConfiguration::reportOnNNPAUnsupportedOps = 0;

  // Run the unknown dimension analysis to help check equality of unknown
  // dimensions at compile time.
  DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

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
    localConfigObject = std::make_unique<JsonConfigObject>();
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
    // Use the reusable applyConfigToOps method.
    configObject->applyConfigToOps(ops, DEVICE_PLACEMENT_KEY,
        [&](llvm::json::Object *jsonObj, mlir::Operation *op) {
          StringRef device = jsonObj->getString(DEVICE_ATTRIBUTE).value();
          op->setAttr(DEVICE_ATTRIBUTE,
              StringAttr::get(module.getContext(), device));
        });
  }

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
  (void)applyAnalysisConversion(module, target, std::move(Patterns1),
      ConversionConfig{.legalizableOps = &legalizedOps1});

  // Call ONNXToZHigh pass for lowering multiple ONNX ops at once to ZHigh.
  // E.g. `onnx.ReLu (onnx.Conv)` to zhigh.Conv.
  RewritePatternSet Patterns2(context);
  getONNXToZHighMultipleOpPatterns(Patterns2);
  (void)applyAnalysisConversion(module, target, std::move(Patterns2),
      ConversionConfig{.legalizableOps = &legalizedOps2});

  // Call ONNXToZHigh pass for lowering a single ONNX op to ZHigh.
  RewritePatternSet Patterns3(context);
  getONNXToZHighOneOpPatterns(Patterns3);
  getONNXToZHighOneOpDynamicallyLegal(&target, &dimAnalysis);
  (void)applyAnalysisConversion(module, target, std::move(Patterns3),
      ConversionConfig{.legalizableOps = &legalizedOps3});

  // Get the legalized ops that will run on the host.
  OpSetType cpuOps = llvm::set_intersection(
      legalizedOps1, llvm::set_intersection(legalizedOps2, legalizedOps3));

  initPlacementHeuristic();
  if (placementHeuristic == QualifyingOps)
    PlaceAllLegalOpsOnNNPA(context, ops, cpuOps);
  else if (placementHeuristic == FasterOps)
    PlaceBeneficialOpsOnNNPA(context, ops, &dimAnalysis, cpuOps);
  else if (placementHeuristic == FasterOpsWSU)
    PlaceBeneficialOpsOnNNPAWithStickUnstick(
        context, module, ops, &dimAnalysis, cpuOps);
  else if (placementHeuristic == MuchFasterOpsWSU)
    PlaceBeneficialOpsOnNNPAWithStickUnstick(context, module, ops, &dimAnalysis,
        cpuOps, /*min factor*/ 3.0, /*significant CPU Factor*/ 2.0,
        /*significant NNPA Factor*/ 8.0);

  // Create a JSON configuration file if required.
  if (!saveConfigFile.empty()) {
    // Save device placement information to a json file by adding to the existing
    // json file an json object of key DEVICE_PLACEMENT_KEY.
    // Each value in the object is added a pair (DEVICE_ATTRIBUTE, value) that
    // denotes the value of DEVICE_ATTRIBUTE in the operation.
    NNPAJsonConfig saveCfg(DEVICE_PLACEMENT_KEY);
    saveCfg.saveConfigToFile(
        ops, saveConfigFile, [&](llvm::json::Object *jsonObj, Operation *op) {
          std::string deviceStr =
              op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE)
                  ? op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE)
                        .getValue()
                        .str()
                  : "";
          jsonObj->insert({DEVICE_ATTRIBUTE, deviceStr});
        });
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
    NNPAPlacementHeuristic placementHeuristic) {
  return std::make_unique<DevicePlacementPass>("", "", placementHeuristic);
}

std::unique_ptr<mlir::Pass> createDevicePlacementPass(
    std::string loadConfigFile, std::string saveConfigFile,
    NNPAPlacementHeuristic placementHeuristic) {
  return std::make_unique<DevicePlacementPass>(
      loadConfigFile, saveConfigFile, placementHeuristic);
}

} // namespace onnx_mlir
