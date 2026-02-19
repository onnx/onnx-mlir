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

const llvm::json::Object *JsonConfigObject::getJsonObject() const {
  return jsonObject.get();
}

llvm::json::Object *JsonConfigObject::getJsonObject() {
  return jsonObject.get();
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

llvm::json::Object *JsonConfigObject::getObject(llvm::StringRef key) {
  if (!jsonObject)
    return nullptr;
  return jsonObject->getObject(key);
}

const llvm::json::Object *JsonConfigObject::getObject(
    llvm::StringRef key) const {
  if (!jsonObject)
    return nullptr;
  return jsonObject->getObject(key);
}

std::optional<llvm::StringRef> JsonConfigObject::getString(
    llvm::StringRef key) const {
  if (!jsonObject)
    return std::nullopt;
  return jsonObject->getString(key);
}

void JsonConfigObject::set(llvm::StringRef key, llvm::json::Value value) {
  if (!jsonObject)
    jsonObject = std::make_unique<llvm::json::Object>();
  (*jsonObject)[key] = std::move(value);
}

bool JsonConfigObject::remove(llvm::StringRef key) {
  if (!jsonObject)
    return false;
  return jsonObject->erase(key);
}

void JsonConfigObject::clear() {
  if (jsonObject)
    jsonObject->clear();
  filePath.clear();
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
    llvm::StringRef arrayKey,
    llvm::function_ref<void(llvm::json::Object *jsonObj, mlir::Operation *op)>
        updateAttrFn) {
  if (!jsonObject || jsonObject->empty())
    return;

  // Get the JSON array for the specified key.
  llvm::json::Array *jsonArr = getArray(arrayKey);
  if (!jsonArr || jsonArr->empty())
    return;

  // Collect operations to work on.
  llvm::DenseSet<mlir::Operation *> workingOps(ops.begin(), ops.end());

  // Process each configuration rule in the JSON array.
  for (llvm::json::Value &v : *jsonArr) {
    llvm::json::Object *vobj = v.getAsObject();
    if (!vobj)
      continue;

    std::optional<llvm::StringRef> nodeType = vobj->getString("node_type");
    std::optional<llvm::StringRef> nodeName = vobj->getString("onnx_node_name");

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
        if (!std::regex_match(
                opNodeName.str(), std::regex(nodeName->str())))
          continue;
      }

      // Apply the callback function when all conditions are satisfied.
      updateAttrFn(vobj, op);
      updatedOps.insert(op);
    }

    // Remove updated ops from working set to avoid processing them again.
    workingOps = llvm::set_difference(workingOps, updatedOps);
  }
}

} // namespace onnx_mlir
