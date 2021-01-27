#pragma once

#include <map>
#include <string>
#include <vector>

#include "mlir/IR/Value.h"
#include "onnx/onnx_pb.h"

namespace onnx_mlir {

void replaceAll(
    std::string &str, const std::string &from, const std::string &to);

std::string legalize_name(std::string name);

/*!
 * A data structure for maintaining mappings from symbol names to symbol values
 * within a single variable scope.
 */
struct VariableScope {
  /*!
   * Create a variable scope.
   * @param identifier name of the variable scope.
   */
  explicit VariableScope(std::string identifier)
      : identifier(std::move(identifier)){};

  /*!
   * Record a symbol name value correspondence.
   * @param name symbol name.
   * @param value symbol value.
   */
  void set(std::string name, mlir::Value value);

  /*!
   * Retrieve the symbol value associated with a name. An assertion failure will
   * occur if symbol is not found.
   * @param name symbol name.
   * @return symbol value.
   */
  mlir::Value get(std::string name) const;

  /*!
   * Check whether symbol exists in the current scope.
   * @param name symbol name.
   * @return whether symbol exists.
   */
  bool contain(std::string name) const;

  /*!
   * Identifier of the current scope, used for debugging and sanity check.
   */
  const std::string identifier;

private:
  /*!
   * A mapping between symbol name and symbol value.
   */
  std::map<std::string, mlir::Value> _nameToValue;
};

/*!
 * A data structure for representing symbol table.
 */
struct SymbolMapping {

  /*!
   *  Get MLIR tensor by onnx tensor name.
   *  @param name onnx tensor name.
   *  @return onnx mlir tensor corresponding to `name`.
   */
  mlir::Value GetTensorByOnnxName(const std::string &name);

  /*!
   *  Add a new mapping from onnx tensor name to MLIR symbol.
   *  @param name onnx tensor name.
   *  @param tensor MLIR Value  pointer.
   */
  void AddMapping(const std::string &name, mlir::Value tensor);

  /*!
   * Check whether a symbol with the specified name exists.
   * @param name symbol name.
   * @return whether a symbol with the sepcified name exists.
   */
  bool ContainKey(std::string name);

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
  std::vector<VariableScope> _scopes;
};

} // namespace onnx_mlir
