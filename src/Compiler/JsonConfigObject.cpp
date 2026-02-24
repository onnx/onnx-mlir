/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- JsonConfigObject.cpp - JSON Config Object ----------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a C++ object to store and manipulate JSON configuration
// data loaded from a file. This is a general-purpose utility that can be
// extended by accelerator plugins.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Compiler/JsonConfigObject.hpp"

namespace onnx_mlir {

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

bool JsonConfigObject::storeToFile(
    const std::string &filePath, unsigned indent) const {
  if (!jsonObject || jsonObject->empty()) {
    llvm::errs() << "Error: Cannot store empty JSON object to file: "
                 << filePath << "\n";
    return false;
  }

  // Open file for writing.
  std::error_code EC;
  llvm::raw_fd_ostream outFile(filePath, EC);
  if (EC) {
    llvm::errs() << "Error: Could not open file for writing: " << filePath
                 << "\n";
    return false;
  }

  // Write JSON with pretty printing.
  llvm::json::OStream jsonOS(outFile, indent);
  jsonOS.value(llvm::json::Value(llvm::json::Object(*jsonObject)));
  outFile << "\n";

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

std::optional<std::string> JsonConfigObject::getCompileOptions() const {
  if (auto opts = getString("compile_options")) {
    return opts->str();
  }
  return std::nullopt;
}


void JsonConfigObject::setCompileOptions(const std::string &options) {
  if (jsonObject) {
    (*jsonObject)["compile_options"] = options;
  }
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

} // namespace onnx_mlir