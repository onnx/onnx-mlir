/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <string>
#include <vector>

#include "mlir/IR/Value.h"
#include "onnx/onnx_pb.h"

namespace onnx_mlir {
/*!
 * A data structure for maintaining mappings from symbol names to symbol values
 * within a single variable scope.
 */
template <typename T>
struct VariableScope {
  /*!
   * Create a variable scope.
   * @param identifier name of the variable scope.
   */
  explicit VariableScope<T>(std::string identifier)
      : identifier(std::move(identifier)){};

  /*!
   * Record a symbol name value correspondence.
   * @param name symbol name.
   * @param value symbol value.
   */
  void set(const std::string &name, T value);

  /*!
   * Retrieve the symbol value associated with a name. An assertion failure will
   * occur if symbol is not found.
   * @param name symbol name.
   * @return symbol value.
   */
  T get(const std::string &name) const;

  /*!
   * Check whether symbol exists in the current scope.
   * @param name symbol name.
   * @return whether symbol exists.
   */
  bool contain(const std::string &name) const;

  /*!
   * Identifier of the current scope, used for debugging and sanity check.
   */
  const std::string identifier;

private:
  /*!
   * A mapping between symbol name and symbol value.
   */
  std::map<std::string, T> _nameToValue;
};

/*!
 * A data structure for representing symbol table.
 */
template <typename T>
struct SymbolMapping {

  /*!
   *  Get MLIR tensor by onnx tensor name.
   *  @param name onnx tensor name.
   *  @return onnx mlir tensor corresponding to `name`.
   */
  T GetTensorByOnnxName(const std::string &name);

  /*!
   *  Add a new mapping from onnx tensor name to MLIR symbol.
   *  @param name onnx tensor name.
   *  @param tensor MLIR Value  pointer.
   */
  void AddMapping(const std::string &name, T tensor);

  /*!
   * Check whether a symbol with the specified name exists.
   * @param name symbol name.
   * @return whether a symbol with the sepcified name exists.
   */
  bool ContainKey(const std::string &name);

  /*!
   * Push a new variable scope with a specified identifier to symbol table.
   * @param identifier identifier for the variable scope.
   */
  void pushScope(const std::string &identifier);

  /*!
   * Pop the outermost variable scope, and checks wether the identifer match up.
   * @param scopeIdentifier identifier for the outermost variable scope, if
   * different from the identifier of the outermost scope, an assertion failure
   * will occur.
   */
  void popScope(const std::string &scopeIdentifier);

private:
  /*!
   *  A list of variable scope, ordered from outermost to innermost.
   */
  std::vector<VariableScope<T>> _scopes;
};


struct NewInitializedTensorMapping : SymbolMapping<onnx::TensorProto> {
/*
  mlir::Value EmitInitializerForInputTensor(
      mlir::Location loc, mlir::OpBuilder &builder, const std::string &name);
*/

  // Get initialized tensor.
  onnx::TensorProto GetInitializedTensor(std::string name) {
    return GetTensorByOnnxName(name);
  }

};

#include "SymbolTable.tpp"

} // namespace onnx_mlir
