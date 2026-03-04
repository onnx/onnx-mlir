/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- NNPAJsonConfigObject.cpp - NNPA JSON Config ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements NNPA-specific JSON configuration functionality.
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <regex>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Accelerators/NNPA/Compiler/NNPAJsonConfigObject.hpp"
#include "src/Compiler/JsonConfigObject.hpp"

using namespace llvm;
using namespace mlir;

namespace onnx_mlir {

// Accessor function to get the global config object with thread-safe
// initialization.
NNPAJsonConfigObject &getGlobalNNPAConfig() {
  static NNPAJsonConfigObject globalNNPAConfig;
  static std::once_flag initFlag;

  std::call_once(initFlag, []() {
    if (!globalNNPAConfig.isLoaded()) {
      JsonConfigObject *globalConfig = &getGlobalOMConfig();
      if (globalConfig->isLoaded())
        globalNNPAConfig.loadFromFile(getGlobalOMConfig().getFilePath());
    }
  });

  return globalNNPAConfig;
}

void NNPAJsonConfigObject::constructTensorInfo(
    Value v, json::Object &tensorInfoObj) {
  // Read info from value and put them into a JSON object as follows:
  // -1 is used for a dynamic dimension.
  // {
  //   "rank": 4,
  //   "type":  "f32"
  //   "dims": {
  //     0: "-1",
  //     1: "3",
  //     2: "5",
  //   },
  // },
  ShapedType tensorType = mlir::dyn_cast<ShapedType>(v.getType());
  if (!tensorType)
    return;

  // Tensor information
  ArrayRef<int64_t> dims = tensorType.getShape();
  // Rank
  tensorInfoObj["rank"] = std::to_string(dims.size());
  // Element type
  Type elemTy = tensorType.getElementType();
  std::string typeStr;
  llvm::raw_string_ostream(typeStr) << elemTy;
  tensorInfoObj["type"] = typeStr;
  // Dimension size
  json::Object dimObj;
  for (uint64_t i = 0; i < dims.size(); ++i) {
    int64_t d = dims[i];
    if (ShapedType::isDynamic(d))
      dimObj[std::to_string(i)] = "-1";
    else
      dimObj[std::to_string(i)] = std::to_string(d);
  }
  tensorInfoObj["dims"] = std::move(dimObj);
}

bool NNPAJsonConfigObject::matchNodeType(mlir::Operation *op, std::regex re) {
  std::string opName = op->getName().getStringRef().str();
  return std::regex_match(opName, re);
}

bool NNPAJsonConfigObject::matchNodeName(mlir::Operation *op, std::regex re) {
  if (auto nameAttr =
          op->getAttrOfType<mlir::StringAttr>(ONNX_NODE_NAME_ATTR)) {
    std::string name = nameAttr.getValue().str();
    return std::regex_match(name, re);
  }
  return false;
}

bool NNPAJsonConfigObject::matchTensorInfo(Value tensor, json::Object *regObj) {
  // clang-format off
  // regObj format
  //   {
  //     "rank": 4, "type":  "f32", "dims": { 0: ">=2", 1: "3", 2: "%32==0", -1:"%64==0"}
  //   },
  // clang-format on

  // Construct a target json object from the tensor.
  json::Object targetObj;
  constructTensorInfo(tensor, targetObj);

  // Match the target object against the regex object.
  bool matched = true;
  for (const auto &kv : *regObj) {
    StringRef k = kv.first;
    if (k.equals_insensitive("rank")) {
      matched &= (targetObj.getString(k) == regObj->getString(k));
    }
    if (k.equals_insensitive("type")) {
      matched &= (targetObj.getString(k) == regObj->getString(k));
    }
    if (!matched)
      break;
  }

  return matched;
}

bool NNPAJsonConfigObject::matchTensorInfo(
    ValueRange tensors, json::Object *regObj) {
  // clang-format off
  // regObj format
  // {
  //   "0": { "rank": 4, "type":  "f32" "dims": { 0: ">=2", 1: "3", 2: "%32==0", -1:"%64==0"} },
  //   "1": { "rank": 4, "type":  "f32" "dims": { 0: ">=2", 1: "3", 2: "%32==0", -1:"%64==0"} },
  // }
  // clang-format on
  int64_t numValues = tensors.size();

  bool matched = true;
  for (const auto &kv : *regObj) {
    StringRef k = kv.first;
    json::Object *v = regObj->getObject(k);
    int id = std::stoi(k.str());
    if (id < 0)
      id += numValues;
    if (id < 0 || id >= numValues) {
      llvm::errs() << "Error: Invalid input/output index.\n";
      matched = false;
      break;
    }
    if (!matchTensorInfo(tensors[id], v)) {
      matched = false;
      break;
    }
  }

  return matched;
}

void NNPAJsonConfigObject::applyConfigToOps(
    llvm::ArrayRef<mlir::Operation *> ops,
    mlir::function_ref<void(json::Object *, mlir::Operation *)> updateAttrFn) {
  if (empty())
    return;

  // Get the nnpa_ops_config array.
  json::Array *opConfigsArr = getArray(OPS_CONFIG_KEY);
  if (!opConfigsArr || opConfigsArr->empty())
    return;

  // Collect operations to work on.
  llvm::DenseSet<mlir::Operation *> workingOps(ops.begin(), ops.end());

  // Process each configuration rule in the nnpa_ops_config array.
  for (json::Value &v : *opConfigsArr) {
    json::Object *configObj = v.getAsObject();
    if (!configObj)
      continue;

    // Get the pattern object.
    json::Object *patternObj = configObj->getObject(PATTERN_KEY);
    if (!patternObj)
      continue;

    // Get the match and rewrite objects.
    json::Object *matchObj = patternObj->getObject(MATCH_KEY);
    json::Object *rewriteObj = patternObj->getObject(REWRITE_KEY);
    if (!matchObj || !rewriteObj)
      continue;

    // Extract matching criteria.
    std::optional<StringRef> nodeTypeStr = matchObj->getString(NODE_TYPE_KEY);
    std::optional<StringRef> onnxNodeNameStr =
        matchObj->getString(ONNX_NODE_NAME_KEY);
    json::Object *inputTensorsObj = matchObj->getObject(INPUTS_KEY);
    json::Object *outputTensorsObj = matchObj->getObject(OUTPUTS_KEY);

    if (!nodeTypeStr) {
      llvm::errs()
          << "Warning: Config entry missing required 'node_type' field\n";
      continue;
    }

    // Create regex patterns for matching with exception handling.
    std::regex nodeTypeRegex;
    std::regex onnxNodeNameRegex;
    bool hasNodeNamePattern = false;
    try {
      nodeTypeRegex = std::regex(nodeTypeStr->str());
      if (onnxNodeNameStr) {
        onnxNodeNameRegex = std::regex(onnxNodeNameStr->str());
        hasNodeNamePattern = true;
      }
    } catch (const std::regex_error &e) {
      llvm::errs() << "Error: Invalid regex pattern in config - " << e.what()
                   << "\n";
      if (onnxNodeNameStr) {
        llvm::errs() << "  node_type: " << nodeTypeStr->str()
                     << ", onnx_node_name: " << onnxNodeNameStr->str() << "\n";
      } else {
        llvm::errs() << "  node_type: " << nodeTypeStr->str() << "\n";
      }
      continue;
    }

    // Find matching operations and apply rewrite.
    llvm::SmallVector<mlir::Operation *> matchedOps;
    for (mlir::Operation *op : workingOps) {
      // Check node type.
      if (!matchNodeType(op, nodeTypeRegex))
        continue;

      // Check onnx_node_name if specified.
      if (hasNodeNamePattern && !matchNodeName(op, onnxNodeNameRegex))
        continue;

      // Check the tensor information.
      ValueRange inputTensors = ValueRange(op->getOperands());
      if (!matchTensorInfo(inputTensors, inputTensorsObj))
        continue;
      ValueRange outputTensors = ValueRange(op->getResults());
      if (!matchTensorInfo(outputTensors, outputTensorsObj))
        continue;

      // Operation matches - apply rewrite.
      updateAttrFn(rewriteObj, op);
      matchedOps.push_back(op);
    }

    // Remove matched operations from working set (first match wins).
    for (mlir::Operation *op : matchedOps) {
      workingOps.erase(op);
    }

    // Stop if no more operations to process.
    if (workingOps.empty())
      break;
  }
}

void NNPAJsonConfigObject::writeOpsConfig(llvm::ArrayRef<mlir::Operation *> ops,
    mlir::function_ref<bool(mlir::Operation *, json::Object &rewrite)>
        buildConfigFn) {
  json::Array opConfigsArray;

  for (mlir::Operation *op : ops) {
    json::Object match;
    json::Object rewrite;

    // Get the operation type.
    std::string nodeType = op->getName().getStringRef().str();
    match[NODE_TYPE_KEY] = nodeType;

    // Get the onnx_node_name if present.
    if (auto nameAttr =
            op->getAttrOfType<mlir::StringAttr>(ONNX_NODE_NAME_ATTR)) {
      match[ONNX_NODE_NAME_KEY] = nameAttr.getValue().str();
    }

    // Get the tensor info from inputs and outputs.
    if (op->getOperands().size() > 0) {
      json::Object inputs;
      for (uint64_t i = 0; i < op->getOperands().size(); ++i) {
        json::Object tensorInfo;
        constructTensorInfo(op->getOperands()[i], tensorInfo);
        inputs[std::to_string(i)] = std::move(tensorInfo);
      }
      match[INPUTS_KEY] = std::move(inputs);
    }
    if (op->getResults().size() > 0) {
      json::Object outputs;
      for (uint64_t i = 0; i < op->getResults().size(); ++i) {
        json::Object tensorInfo;
        constructTensorInfo(op->getResults()[i], tensorInfo);
        outputs[std::to_string(i)] = std::move(tensorInfo);
      }
      match[OUTPUTS_KEY] = std::move(outputs);
    }

    // Let the callback build the rewrite object.
    if (!buildConfigFn(op, rewrite))
      continue;

    // Build the pattern object.
    json::Object pattern;
    pattern[MATCH_KEY] = std::move(match);
    pattern[REWRITE_KEY] = std::move(rewrite);

    // Build the config object.
    json::Object config;
    config[PATTERN_KEY] = std::move(pattern);

    opConfigsArray.push_back(std::move(config));
  }

  // Store the nnpa_ops_config array in the JSON object.
  json::Object *jsonObj = getJsonObject();
  if (jsonObj) {
    (*jsonObj)[OPS_CONFIG_KEY] = std::move(opConfigsArray);
  }
}

} // namespace onnx_mlir
