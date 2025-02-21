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

#include <regex>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/DevicePlacementHeuristic.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "device-placement"

using namespace mlir;
using namespace onnx_mlir;

namespace {

// Global object to ease error reporting, it consumes errors and crash the
// application with a meaningful message.
static llvm::ExitOnError ExitOnErr;

struct DevicePlacementPass
    : public PassWrapper<DevicePlacementPass, OperationPass<ModuleOp>> {
  using OpSetType = DenseSet<Operation *>;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DevicePlacementPass)

  DevicePlacementPass() = default;
  DevicePlacementPass(const DevicePlacementPass &pass)
      : PassWrapper<DevicePlacementPass, OperationPass<ModuleOp>>() {
    this->placementHeuristic = QualifyingOps;
  }
  DevicePlacementPass(std::string loadConfigFile, std::string saveConfigFile,
      NNPAPlacementHeuristic placementHeuristic) {
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

  // JSON keys.
  std::string DEVICE_KEY = "device";
  std::string DEVICE_PLACEMENT_KEY = "device_placement";
  std::string NODE_TYPE_KEY = "node_type";
  std::string ONNX_NODE_NAME_KEY = "onnx_node_name";

  // Exclude these operations from device placement.
  bool isExcludedOp(Operation *op) {
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return true;
    // No annotation for these ops.
    if (isa<ONNXEntryPointOp, ONNXReturnOp, ONNXConstantOp>(op))
      return true;
    return false;
  }

  // Functions to load/save device placement from/to a JSON file.
  // JSON file example:
  // ```json
  // {
  //   "device_placement": [
  //     {
  //       "device": "cpu",
  //       "node_type": "onnx.Relu",
  //       "onnx_node_name": "Relu_[1,2]"
  //     },
  //     {
  //       "device": "nnpa",
  //       "node_type": "onnx.Sigmoid",
  //       "onnx_node_name": ".*"
  //     }
  //   ]
  // }
  // ```
  void loadConfigFromJSONFile();
  void saveConfigToJSONFile();
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

  // Cost model and user configuration file go here if it's given.
  // (Reserved for cost model and user configuration file)
  if (!loadConfigFile.empty())
    loadConfigFromJSONFile();

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
  if (!saveConfigFile.empty())
    saveConfigToJSONFile();
}

void DevicePlacementPass::loadConfigFromJSONFile() {
  auto Buf = ExitOnErr(errorOrToExpected(
      llvm::MemoryBuffer::getFile(loadConfigFile, /*bool IsText=*/true,
          /*RequiresNullTerminator=*/false)));
  auto jsonFile = ExitOnErr(llvm::json::parse(Buf->getBuffer()));
  llvm::json::Object *jsonContent = jsonFile.getAsObject();
  llvm::json::Array *jsonArr = jsonContent->getArray(DEVICE_PLACEMENT_KEY);
  if (!jsonArr || jsonArr->empty())
    return;

  // Collect operations to work on.
  OpSetType workingOps(ops.begin(), ops.end());
  // Go over operations in the JSON and find matched operation in the IR.
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
      // Match operation.
      if (!std::regex_match(opNodeType.str(), std::regex(nodeType.str())))
        continue;
      if (!std::regex_match(opNodeName.str(), std::regex(nodeName.str())))
        continue;
      // Set device.
      op->setAttr(
          DEVICE_ATTRIBUTE, StringAttr::get(module.getContext(), device));
      updatedOps.insert(op);
    }
    // To reduce complexity, once an operation is assigned a device, we remove
    // it from the set workingOps.
    workingOps = llvm::set_difference(workingOps, updatedOps);
  }
}

void DevicePlacementPass::saveConfigToJSONFile() {
  // Parsing the module to JSON object.
  llvm::json::Array jsonArr;
  for (Operation *op : ops) {
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
  }
  llvm::json::Object jsonContent{
      {DEVICE_PLACEMENT_KEY, llvm::json::Value(std::move(jsonArr))}};

  // Exporting the JSON object to a file.
  std::error_code EC;
  llvm::raw_fd_ostream jsonOS(saveConfigFile, EC);
  if (EC)
    report_fatal_error(
        "Error saving device placement json file : " + StringRef(EC.message()));
  jsonOS << llvm::json::Value(std::move(jsonContent)) << "\n";
  jsonOS.close();
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
    std::string loadConfigFile, std::string saveConfigFile,
    NNPAPlacementHeuristic placementHeuristic) {
  return std::make_unique<DevicePlacementPass>(
      loadConfigFile, saveConfigFile, placementHeuristic);
}

} // namespace onnx_mlir
