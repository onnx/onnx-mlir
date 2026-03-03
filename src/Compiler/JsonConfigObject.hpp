/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- JsonConfigObject.hpp - JSON Config Object ----------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file defines a C++ object to store and manipulate JSON configuration
// data loaded from a file. This is a general-purpose JSON config utility
// used by the compiler and accelerator plugins.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_COMPILER_JSON_CONFIG_OBJECT_H
#define ONNX_MLIR_COMPILER_JSON_CONFIG_OBJECT_H

#include <memory>
#include <string>

#include "llvm/Support/JSON.h"

namespace onnx_mlir {

/// A general-purpose C++ object that stores JSON configuration data loaded
/// from a file. Provides basic methods to load, access, and query JSON data.
///
/// This is a base class that can be extended by accelerator plugins to add
/// domain-specific functionality.

/// The base JSON configuration supports the following keys:
/// @code{.json}
/// {
///   "compile_options": ["-O3", "-march=z16"],
/// }
/// @endcode

class JsonConfigObject {
public:
  /// Constructor - creates an empty JSON config object.
  JsonConfigObject();

  /// Destructor.
  virtual ~JsonConfigObject();

  /// Load JSON content from a file.
  /// @param filePath Path to the JSON file to load.
  /// @return true if successful, false otherwise.
  bool loadFromFile(const std::string &filePath);

  /// Store JSON content to a file.
  /// @param filePath Path to the JSON file to write.
  /// @param indent Indentation size for pretty printing (default: 2).
  /// @return true if successful, false otherwise.
  bool storeToFile(const std::string &filePath, unsigned indent = 2) const;

  /// Check if the config object is empty.
  /// @return true if empty, false otherwise.
  bool empty() const;

  /// Get a JSON array by key.
  /// @param key The key to look up.
  /// @return Pointer to the JSON array, or nullptr if not found.
  llvm::json::Array *getArray(llvm::StringRef key);

  /// Get a JSON array by key (read-only).
  /// @param key The key to look up.
  /// @return Pointer to the JSON array, or nullptr if not found.
  const llvm::json::Array *getArray(llvm::StringRef key) const;

  /// Get a string value by key.
  /// @param key The key to look up.
  /// @return Optional string value.
  std::optional<llvm::StringRef> getString(llvm::StringRef key) const;

  /// Get compile options from the config.
  /// @return true if successful, false otherwise.
  bool getCompileOptions(std::vector<std::string> &args) const;

  /// Dump the JSON content to stdout for debugging.
  /// @param indent Indentation size for pretty printing (default: 2).
  void dump(unsigned indent = 2) const;

  /// Get the file path that was last loaded.
  /// @return The file path string.
  const std::string &getFilePath() const { return filePath; }

  /// Check if the config file is loaded.
  /// @return true if loaded, false otherwise.
  bool isLoaded() const { return fileIsLoaded; }

  // JSON key constants.
  static constexpr const char *COMPILE_OPTIONS_KEY = "compile_options";

protected:
  /// Get access to the underlying JSON object for derived classes.
  /// @return Pointer to the JSON object, or nullptr if not loaded.
  llvm::json::Object *getJsonObject() { return jsonObject.get(); }

  /// Get read-only access to the underlying JSON object for derived classes.
  /// @return Pointer to the JSON object, or nullptr if not loaded.
  const llvm::json::Object *getJsonObject() const { return jsonObject.get(); }

private:
  /// The underlying JSON object.
  std::unique_ptr<llvm::json::Object> jsonObject;

  /// The file path that was last loaded.
  std::string filePath;

  /// Track if the config file is loaded or not.
  bool fileIsLoaded = false;
};

/// Get the global configuration object.
JsonConfigObject &getGlobalOMConfig();

} // namespace onnx_mlir

#endif // ONNX_MLIR_COMPILER_JSON_CONFIG_OBJECT_H
