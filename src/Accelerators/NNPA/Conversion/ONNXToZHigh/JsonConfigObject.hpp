/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- JsonConfigObject.hpp - JSON Config Object ----------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file defines a C++ object to store and manipulate JSON configuration
// data loaded from a file.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_JSON_CONFIG_OBJECT_H
#define ONNX_MLIR_JSON_CONFIG_OBJECT_H

#include <memory>
#include <string>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/JSON.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace onnx_mlir {

/// A C++ object that stores JSON configuration data loaded from a file.
/// Provides methods to load, access, modify, and save JSON data.
class JsonConfigObject {
public:
  /// Constructor - creates an empty JSON config object.
  JsonConfigObject();

  /// Destructor.
  ~JsonConfigObject();

  /// Load JSON content from a file.
  /// @param filePath Path to the JSON file to load
  /// @return true if successful, false otherwise
  bool loadFromFile(const std::string &filePath);


  /// Check if the config object is empty.
  /// @return true if empty, false otherwise
  bool empty() const;

  /// Get the underlying JSON object (read-only).
  /// @return Pointer to the JSON object, or nullptr if empty
  const llvm::json::Object *getJsonObject() const;

  /// Get the underlying JSON object (mutable).
  /// @return Pointer to the JSON object, or nullptr if empty
  llvm::json::Object *getJsonObject();

  /// Get a JSON array by key.
  /// @param key The key to look up
  /// @return Pointer to the JSON array, or nullptr if not found
  llvm::json::Array *getArray(llvm::StringRef key);

  /// Get a JSON array by key (read-only).
  /// @param key The key to look up
  /// @return Pointer to the JSON array, or nullptr if not found
  const llvm::json::Array *getArray(llvm::StringRef key) const;

  /// Get a JSON object by key.
  /// @param key The key to look up
  /// @return Pointer to the JSON object, or nullptr if not found
  llvm::json::Object *getObject(llvm::StringRef key);

  /// Get a JSON object by key (read-only).
  /// @param key The key to look up
  /// @return Pointer to the JSON object, or nullptr if not found
  const llvm::json::Object *getObject(llvm::StringRef key) const;

  /// Get a string value by key.
  /// @param key The key to look up
  /// @return Optional string value
  std::optional<llvm::StringRef> getString(llvm::StringRef key) const;

  /// Set a value for a key.
  /// @param key The key to set
  /// @param value The JSON value to set
  void set(llvm::StringRef key, llvm::json::Value value);

  /// Remove a key from the JSON object.
  /// @param key The key to remove
  /// @return true if the key was found and removed, false otherwise
  bool remove(llvm::StringRef key);

  /// Clear all content.
  void clear();

  /// Dump the JSON content to stdout for debugging.
  /// @param indent Indentation size for pretty printing (default: 2)
  void dump(unsigned indent = 2) const;

  /// Apply configuration from a JSON array to operations using a callback.
  /// This method has the same signature as NNPAJsonConfig::loadConfigFromFile
  /// @param ops Array of operations to process
  /// @param arrayKey The key of the JSON array containing configuration rules
  /// @param updateAttrFn Callback function to update operation attributes
  void applyConfigToOps(llvm::ArrayRef<mlir::Operation *> ops,
      llvm::StringRef arrayKey,
      mlir::function_ref<void(llvm::json::Object *, mlir::Operation *)>
          updateAttrFn);

  /// Get the file path that was last loaded.
  /// @return The file path, or empty string if not loaded from file
  const std::string &getFilePath() const { return filePath; }

private:
  /// The underlying JSON object.
  std::unique_ptr<llvm::json::Object> jsonObject;

  /// The file path that was last loaded.
  std::string filePath;
};

/// Get the global NNPA configuration object.
JsonConfigObject &getGlobalNNPAConfig();

} // namespace onnx_mlir

#endif // ONNX_MLIR_JSON_CONFIG_OBJECT_H
