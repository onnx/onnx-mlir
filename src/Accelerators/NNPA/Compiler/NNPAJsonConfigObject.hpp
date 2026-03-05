/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- NNPAJsonConfigObject.hpp - NNPA JSON Config ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file defines an NNPA-specific extension of JsonConfigObject that adds
// functionality for device placement and quantization configuration.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NNPA_JSON_CONFIG_OBJECT_H
#define ONNX_MLIR_NNPA_JSON_CONFIG_OBJECT_H

#include <regex>

#include "mlir/IR/Operation.h"
#include "src/Compiler/JsonConfigObject.hpp"
#include "llvm/ADT/STLFunctionalExtras.h"

using namespace mlir;

namespace onnx_mlir {

/// NNPA-specific extension of JsonConfigObject that handles operation
/// configuration for device placement and quantization.
///
/// The JSON configuration uses a format with the following structure:
/// @code{.json}
/// {
///   "compile_options": ["-O3", "-march=z16"],
///   "nnpa_ops_config": [
///     {
///       "pattern": {
///         "match": {
///           "node_type": "onnx.MatMul",
///           "onnx_node_name": "MatMul_1"
///         },
///         "rewrite": {
///           "device": "NNPA",
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
/// - "device": Device placement ("CPU", "NNPA", or empty string)
/// - "quantize": Boolean flag for quantization (true/false)
///
/// Multiple configurations can be specified in the "nnpa_ops_config" array.
/// Configurations are processed in order, and the first matching pattern wins.
class NNPAJsonConfigObject : public JsonConfigObject {
public:
  /// Constructor - creates an empty NNPA JSON config object.
  NNPAJsonConfigObject() = default;

  /// Destructor.
  ~NNPAJsonConfigObject() override = default;

  /// Apply configuration from a json file to operations using a callback.
  /// This method handles the "nnpa_ops_config" format where each config has
  /// "pattern.match" (criteria) and "pattern.rewrite" (attributes) sections.
  /// @param ops Array of operations to process.
  /// @param updateAttrFn Callback function to update operation attributes
  ///                     with the rewrite object.
  void applyConfigToOps(llvm::ArrayRef<mlir::Operation *> ops,
      mlir::function_ref<void(llvm::json::Object *, mlir::Operation *)>
          updateAttrFn);

  /// Build operations configuration in the JSON object.
  /// @param ops Array of operations to process.
  /// @param buildConfigFn Callback to build the rewrite object for each op.
  ///                      Returns true if the operation should be included.
  void writeOpsConfig(llvm::ArrayRef<mlir::Operation *> ops,
      mlir::function_ref<bool(mlir::Operation *, llvm::json::Object &rewrite)>
          buildConfigFn);

  // JSON key constants for NNPA configuration.
  // TODO: define a JSON schema file for validation.
  static constexpr const char *OPS_CONFIG_KEY = "nnpa_ops_config";
  static constexpr const char *PATTERN_KEY = "pattern";
  // Keys inside `pattern'
  static constexpr const char *MATCH_KEY = "match";
  static constexpr const char *REWRITE_KEY = "rewrite";
  // Keys inside `match`.
  static constexpr const char *NODE_TYPE_KEY = "node_type";
  static constexpr const char *ONNX_NODE_NAME_KEY = "onnx_node_name";
  static constexpr const char *INPUTS_KEY = "inputs";
  static constexpr const char *OUTPUTS_KEY = "outputs";
  // Keys inside `rewrite`.
  static constexpr const char *DEVICE_KEY = "device";
  static constexpr const char *QUANTIZE_KEY = "quantize";

  // Attributes in the operations.
  static constexpr const char *ONNX_NODE_NAME_ATTR = ONNX_NODE_NAME_KEY;
  static constexpr const char *DEVICE_ATTR = DEVICE_KEY;
  static constexpr const char *QUANTIZE_ATTR = QUANTIZE_KEY;

private:
  void constructTensorInfo(mlir::Value v, llvm::json::Object &tensorInfoObj);
  bool matchNodeType(mlir::Operation *op, std::regex re);
  bool matchNodeName(mlir::Operation *op, std::regex re);
  bool matchTensorInfo(mlir::Value tensor, llvm::json::Object *patternObj);
  bool matchTensorInfo(mlir::ValueRange tensors, llvm::json::Object *patternObj);
};

/// Get the global NNPA configuration object.
NNPAJsonConfigObject &getGlobalNNPAConfig();

} // namespace onnx_mlir

#endif // ONNX_MLIR_NNPA_JSON_CONFIG_OBJECT_H
