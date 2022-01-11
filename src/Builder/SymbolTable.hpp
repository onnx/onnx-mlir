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
  explicit VariableScope<T>(std::string _identifier)
      : identifier(std::move(_identifier)), nameToValue(){};

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

  const std::string &getIdentifier() const { return identifier; }

private:
  /*!
   * Identifier of the current scope, used for debugging and safety check.
   */
  const std::string identifier;

  /*!
   * A mapping between symbol name and symbol value.
   */
  std::map<std::string, T> nameToValue;
};

/*!
 * A data structure for representing symbol table.
 */
template <typename T>
struct SymbolMapping {
  SymbolMapping() = default;

  /*!
   *  Get MLIR tensor by onnx tensor name.
   *  @param name onnx tensor name.
   *  @return onnx mlir tensor corresponding to `name`.
   */
  T GetTensorByOnnxName(const std::string &name) const;

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
  bool ContainKey(const std::string &name) const;

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
  std::vector<VariableScope<T>> scopes;
};

/**
 * Template definition for member functions
 * Needed in include file for template instantialization
 */

template <typename T>
T SymbolMapping<T>::GetTensorByOnnxName(const std::string &name) const {
  for (const auto &scope : scopes)
    if (scope.contain(name))
      return scope.get(name);
  llvm_unreachable("Tensor not found");
}

template <typename T>
void SymbolMapping<T>::AddMapping(const std::string &name, T tensor) {
  assert(!scopes.empty());
  assert(!scopes.back().contain(name) && "Tensor already exists.");
  scopes.back().set(name, tensor);
}

template <typename T>
bool SymbolMapping<T>::ContainKey(const std::string &name) const {
  return llvm::any_of(scopes,
      [name](const VariableScope<T> &scope) { return scope.contain(name); });
}

template <typename T>
void SymbolMapping<T>::pushScope(const std::string &identifier) {
  scopes.emplace_back(VariableScope<T>(identifier));
}

template <typename T>
void SymbolMapping<T>::popScope(const std::string &scopeIdentifier) {
  assert(scopes.back().getIdentifier() == scopeIdentifier);
  scopes.pop_back();
}

template <typename T>
void VariableScope<T>::set(const std::string &name, T val) {
  assert(nameToValue.count(name) == 0 && "duplicate key in symbol table");
  nameToValue.emplace(name, val);
}

template <typename T>
T VariableScope<T>::get(const std::string &name) const {
  return nameToValue.at(name);
}

template <typename T>
bool VariableScope<T>::contain(const std::string &name) const {
  return nameToValue.count(name) > 0;
}

} // namespace onnx_mlir
