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

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/JSON.h"

using namespace mlir;

namespace onnx_mlir {

/// A C++ object that stores JSON configuration data loaded from a file.
/// Provides methods to load, access, modify, and save JSON data.
///
/// The JSON configuration uses a unified format with the following structure:
/// @code{.json}
/// {
///   "ops_config": [
///     {
///       "pattern": {
///         "match": {
///           "node_type": "onnx.MatMul",
///           "onnx_node_name": "MatMul_1"
///         },
///         "rewrite": {
///           "device": "nnpa",
///           "quantize": true
///         }
///       }
///     }
///   ]
/// }
/// @endcode
///
/// The "match" section specifies criteria for selecting operations:
/// - "node_type": Operation type (supports regex patterns like "onnx.*")
/// - "onnx_node_name": Optional operation name (supports regex patterns)
///
/// The "rewrite" section specifies attributes to apply to matched operations:
/// - "device": Device placement ("cpu", "nnpa", or empty string)
/// - "quantize": Boolean flag for quantization (true/false)
///
/// Multiple configurations can be specified in the "ops_config" array.
/// Configurations are processed in order, and the first matching pattern wins.
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

  /// Get a JSON array by key.
  /// @param key The key to look up
  /// @return Pointer to the JSON array, or nullptr if not found
  llvm::json::Array *getArray(llvm::StringRef key);

  /// Get a JSON array by key (read-only).
  /// @param key The key to look up
  /// @return Pointer to the JSON array, or nullptr if not found
  const llvm::json::Array *getArray(llvm::StringRef key) const;

  /// Get a string value by key.
  /// @param key The key to look up
  /// @return Optional string value
  std::optional<llvm::StringRef> getString(llvm::StringRef key) const;

  /// Dump the JSON content to stdout for debugging.
  /// @param indent Indentation size for pretty printing (default: 2)
  void dump(unsigned indent = 2) const;

  /// Apply configuration from unified format to operations using a callback.
  /// This method handles the "ops_config" format where each config has
  /// "pattern.matching" (criteria) and "pattern.rewrite" (attributes) sections.
  /// @param ops Array of operations to process
  /// @param updateAttrFn Callback function to update operation attributes
  ///                     with the rewrite object
  void applyConfigToOps(llvm::ArrayRef<mlir::Operation *> ops,
      mlir::function_ref<void(llvm::json::Object *, mlir::Operation *)>
          updateAttrFn);

  /// Write operations configuration to a JSON file in unified format.
  /// @param ops Array of operations to save
  /// @param filePath Path to the output JSON file
  /// @param buildConfigFn Callback to build match and rewrite objects for each op
  ///                      Returns true if the operation should be included
  /// @return true if successful, false otherwise
  bool writeOpsConfig(llvm::ArrayRef<mlir::Operation *> ops,
      const std::string &filePath,
      mlir::function_ref<bool(mlir::Operation *, llvm::json::Object &match,
          llvm::json::Object &rewrite)>
          buildConfigFn);

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
