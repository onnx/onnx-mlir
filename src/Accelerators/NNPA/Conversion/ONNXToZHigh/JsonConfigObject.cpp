/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- JsonConfigObject.cpp - JSON Config Object ----------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a C++ object to store and manipulate JSON configuration
// data loaded from a file.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/JsonConfigObject.hpp"

#include <regex>

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace onnx_mlir {

// Global NNPA configuration object.
static JsonConfigObject globalNNPAConfig;

// Accessor function to get the global config object.
JsonConfigObject &getGlobalNNPAConfig() { return globalNNPAConfig; }

JsonConfigObject::JsonConfigObject()
    : jsonObject(std::make_unique<llvm::json::Object>()), filePath("") {}

JsonConfigObject::~JsonConfigObject() = default;

bool JsonConfigObject::loadFromFile(const std::string &filePath) {
  // Try to load the file.
  auto bufferOrError = llvm::MemoryBuffer::getFile(
      filePath, /*bool IsText=*/true, /*RequiresNullTerminator=*/false);

  if (!bufferOrError) {
    llvm::errs() << "Error: Could not open file: " << filePath << "\n";
    return false;
  }

  // Parse the JSON content.
  auto jsonOrError = llvm::json::parse(bufferOrError.get()->getBuffer());
  if (!jsonOrError) {
    llvm::errs() << "Error: Failed to parse JSON from file: " << filePath
                 << "\n";
    llvm::errs() << "Parse error: " << toString(jsonOrError.takeError())
                 << "\n";
    return false;
  }

  // Extract the JSON object.
  llvm::json::Object *parsedObject = jsonOrError->getAsObject();
  if (!parsedObject) {
    llvm::errs() << "Error: JSON root is not an object in file: " << filePath
                 << "\n";
    return false;
  }

  // Store the parsed JSON object.
  jsonObject = std::make_unique<llvm::json::Object>(std::move(*parsedObject));
  this->filePath = filePath;

  return true;
}

bool JsonConfigObject::empty() const {
  return !jsonObject || jsonObject->empty();
}

llvm::json::Array *JsonConfigObject::getArray(llvm::StringRef key) {
  if (!jsonObject)
    return nullptr;
  return jsonObject->getArray(key);
}

const llvm::json::Array *JsonConfigObject::getArray(llvm::StringRef key) const {
  if (!jsonObject)
    return nullptr;
  return jsonObject->getArray(key);
}

std::optional<llvm::StringRef> JsonConfigObject::getString(
    llvm::StringRef key) const {
  if (!jsonObject)
    return std::nullopt;
  return jsonObject->getString(key);
}

void JsonConfigObject::dump(unsigned indent) const {
  if (!jsonObject || jsonObject->empty()) {
    llvm::outs() << "JsonConfigObject is empty\n";
    return;
  }

  llvm::outs() << "JsonConfigObject contents";
  if (!filePath.empty()) {
    llvm::outs() << " (loaded from: " << filePath << ")";
  }
  llvm::outs() << ":\n";

  // Use LLVM's JSON pretty printer.
  llvm::json::OStream jsonOS(llvm::outs(), indent);
  jsonOS.value(llvm::json::Value(llvm::json::Object(*jsonObject)));
  llvm::outs() << "\n";
}

void JsonConfigObject::applyConfigToOps(llvm::ArrayRef<mlir::Operation *> ops,
    mlir::function_ref<void(
        llvm::json::Object *rewriteObj, mlir::Operation *op)>
        updateAttrFn) {
  if (!jsonObject || jsonObject->empty())
    return;

  // Get the ops_config array.
  llvm::json::Array *opConfigsArr = getArray(OPS_CONFIG_KEY);
  if (!opConfigsArr || opConfigsArr->empty())
    return;

  // Collect operations to work on.
  llvm::DenseSet<mlir::Operation *> workingOps(ops.begin(), ops.end());

  // Process each configuration rule in the ops_config array.
  for (llvm::json::Value &v : *opConfigsArr) {
    llvm::json::Object *configObj = v.getAsObject();
    if (!configObj)
      continue;

    // Get the pattern object.
    llvm::json::Object *patternObj = configObj->getObject(PATTERN_KEY);
    if (!patternObj)
      continue;

    // Get match and rewrite objects.
    llvm::json::Object *matchObj = patternObj->getObject(MATCH_KEY);
    llvm::json::Object *rewriteObj = patternObj->getObject(REWRITE_KEY);

    if (!matchObj || !rewriteObj)
      continue;

    // Extract match criteria.
    std::optional<llvm::StringRef> nodeType =
        matchObj->getString(NODE_TYPE_KEY);
    std::optional<llvm::StringRef> nodeName =
        matchObj->getString(ONNX_NODE_NAME_KEY);

    if (!nodeType)
      continue;

    llvm::DenseSet<mlir::Operation *> updatedOps;
    for (mlir::Operation *op : workingOps) {
      // Match node type using regex.
      llvm::StringRef opNodeType = op->getName().getStringRef();
      if (!std::regex_match(opNodeType.str(), std::regex(nodeType->str())))
        continue;

      // Match node name if specified.
      if (nodeName.has_value()) {
        llvm::StringRef opNodeName =
            op->getAttrOfType<mlir::StringAttr>("onnx_node_name")
                ? op->getAttrOfType<mlir::StringAttr>("onnx_node_name")
                      .getValue()
                : "";
        if (!std::regex_match(opNodeName.str(), std::regex(nodeName->str())))
          continue;
      }

      // Apply the callback function with the rewrite object when all
      // conditions are satisfied.
      updateAttrFn(rewriteObj, op);
      updatedOps.insert(op);
    }

    // Remove updated ops from working set to avoid processing them again.
    workingOps = llvm::set_difference(workingOps, updatedOps);
  }
}

bool JsonConfigObject::writeOpsConfig(llvm::ArrayRef<mlir::Operation *> ops,
    const std::string &filePath,
    mlir::function_ref<bool(mlir::Operation *, llvm::json::Object &match,
        llvm::json::Object &rewrite)>
        buildConfigFn) {
  llvm::json::Array opConfigsArray;

  for (mlir::Operation *op : ops) {
    llvm::json::Object match;
    llvm::json::Object rewrite;

    // Call the callback to build match and rewrite objects.
    if (!buildConfigFn(op, match, rewrite))
      continue;

    // Add node_type to match.
    std::string nodeType = op->getName().getStringRef().str();
    match[JsonConfigObject::NODE_TYPE_KEY] = nodeType;

    // Add onnx_node_name to match if present.
    if (auto nodeNameAttr =
            op->getAttrOfType<mlir::StringAttr>("onnx_node_name")) {
      match[JsonConfigObject::ONNX_NODE_NAME_KEY] =
          nodeNameAttr.getValue().str();
    }

    // Build the pattern structure.
    llvm::json::Object pattern;
    pattern[MATCH_KEY] = std::move(match);
    pattern[REWRITE_KEY] = std::move(rewrite);

    llvm::json::Object config;
    config[PATTERN_KEY] = std::move(pattern);

    opConfigsArray.push_back(std::move(config));
  }

  // Write the configuration to file.
  std::error_code EC;
  llvm::raw_fd_ostream outFile(filePath, EC);
  if (EC) {
    llvm::errs() << "Error: Could not open file for writing: " << filePath
                 << "\n";
    return false;
  }

  llvm::json::Object rootObj;
  rootObj[OPS_CONFIG_KEY] = std::move(opConfigsArray);

  llvm::json::OStream jsonOS(outFile, 2);
  jsonOS.value(llvm::json::Value(std::move(rootObj)));
  outFile << "\n";

  return true;
}

} // namespace onnx_mlir
