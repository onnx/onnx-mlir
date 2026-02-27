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
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Accelerators/NNPA/Compiler/NNPAJsonConfigObject.hpp"
#include "src/Compiler/JsonConfigObject.hpp"

namespace onnx_mlir {

// Accessor function to get the global config object with thread-safe
// initialization.
NNPAJsonConfigObject &getGlobalNNPAConfig() {
  static NNPAJsonConfigObject globalNNPAConfig;
  static std::once_flag initFlag;

  std::call_once(initFlag, []() {
    if (!globalNNPAConfig.isLoaded()) {
      globalNNPAConfig.loadFromFile(getGlobalOMConfig().getFilePath());
    }
  });

  return globalNNPAConfig;
}

void NNPAJsonConfigObject::applyConfigToOps(
    llvm::ArrayRef<mlir::Operation *> ops,
    mlir::function_ref<void(llvm::json::Object *, mlir::Operation *)>
        updateAttrFn) {
  if (empty())
    return;

  // Get the nnpa_ops_config array.
  llvm::json::Array *opConfigsArr = getArray(OPS_CONFIG_KEY);
  if (!opConfigsArr || opConfigsArr->empty())
    return;

  // Collect operations to work on.
  llvm::DenseSet<mlir::Operation *> workingOps(ops.begin(), ops.end());

  // Process each configuration rule in the nnpa_ops_config array.
  for (llvm::json::Value &v : *opConfigsArr) {
    llvm::json::Object *configObj = v.getAsObject();
    if (!configObj)
      continue;

    // Get the pattern object.
    llvm::json::Object *patternObj = configObj->getObject(PATTERN_KEY);
    if (!patternObj)
      continue;

    // Get the match and rewrite objects.
    llvm::json::Object *matchObj = patternObj->getObject(MATCH_KEY);
    llvm::json::Object *rewriteObj = patternObj->getObject(REWRITE_KEY);
    if (!matchObj || !rewriteObj)
      continue;

    // Extract matching criteria.
    auto nodeType = matchObj->getString(NODE_TYPE_KEY);
    auto onnxNodeName = matchObj->getString(ONNX_NODE_NAME_KEY);

    if (!nodeType) {
      llvm::errs()
          << "Warning: Config entry missing required 'node_type' field\n";
      continue;
    }

    // Create regex patterns for matching with exception handling.
    std::regex nodeTypeRegex;
    std::regex onnxNodeNameRegex;
    bool hasNodeNamePattern = false;

    try {
      nodeTypeRegex = std::regex(nodeType->str());

      if (onnxNodeName) {
        onnxNodeNameRegex = std::regex(onnxNodeName->str());
        hasNodeNamePattern = true;
      }
    } catch (const std::regex_error &e) {
      llvm::errs() << "Error: Invalid regex pattern in config - " << e.what()
                   << "\n";
      if (onnxNodeName) {
        llvm::errs() << "  node_type: " << nodeType->str()
                     << ", onnx_node_name: " << onnxNodeName->str() << "\n";
      } else {
        llvm::errs() << "  node_type: " << nodeType->str() << "\n";
      }
      continue;
    }

    // Find matching operations and apply rewrite.
    llvm::SmallVector<mlir::Operation *> matchedOps;
    for (mlir::Operation *op : workingOps) {
      // Check node type.
      std::string opName = op->getName().getStringRef().str();
      if (!std::regex_match(opName, nodeTypeRegex))
        continue;

      // Check onnx_node_name if specified.
      if (hasNodeNamePattern) {
        if (auto nameAttr =
                op->getAttrOfType<mlir::StringAttr>(ONNX_NODE_NAME_ATTR)) {
          std::string name = nameAttr.getValue().str();
          if (!std::regex_match(name, onnxNodeNameRegex))
            continue;
        } else {
          continue; // No name attribute, skip.
        }
      }

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
    mlir::function_ref<bool(mlir::Operation *, llvm::json::Object &rewrite)>
        buildConfigFn) {
  llvm::json::Array opConfigsArray;

  for (mlir::Operation *op : ops) {
    llvm::json::Object match;
    llvm::json::Object rewrite;

    // Get the operation type.
    std::string nodeType = op->getName().getStringRef().str();
    match[NNPAJsonConfigObject::NODE_TYPE_KEY] = nodeType;

    // Get the onnx_node_name if present.
    if (auto nameAttr =
            op->getAttrOfType<mlir::StringAttr>(ONNX_NODE_NAME_ATTR)) {
      match[NNPAJsonConfigObject::ONNX_NODE_NAME_KEY] =
          nameAttr.getValue().str();
    }

    // Let the callback build the rewrite object.
    if (!buildConfigFn(op, rewrite))
      continue;

    // Build the pattern object.
    llvm::json::Object pattern;
    pattern[MATCH_KEY] = std::move(match);
    pattern[REWRITE_KEY] = std::move(rewrite);

    // Build the config object.
    llvm::json::Object config;
    config[PATTERN_KEY] = std::move(pattern);

    opConfigsArray.push_back(std::move(config));
  }

  // Store the nnpa_ops_config array in the JSON object.
  llvm::json::Object *jsonObj = getJsonObject();
  if (jsonObj) {
    (*jsonObj)[OPS_CONFIG_KEY] = std::move(opConfigsArray);
  }
}

} // namespace onnx_mlir
