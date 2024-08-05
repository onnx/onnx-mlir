/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_MLIR_SYMBOL_TABLE_H
#define ONNX_MLIR_SYMBOL_TABLE_H

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnx_mlir {
/*!
 * A data structure for maintaining mappings from symbol names to objects
 * within a single variable scope.
 */
template <typename T>
struct VariableScope {
  /*!
   * Create a variable scope.
   * @param identifier name of the variable scope.
   */
  explicit VariableScope<T>(std::string _identifier)
      : identifier(std::move(_identifier)), nameToObject(){};

  /*!
   * Record a symbol name object correspondence.
   * @param name symbol name.
   * @param object object.
   */
  void set(const std::string &name, T object);

  /*!
   * Retrieve pointer to the object associated with a symbol name, or nullptr
   * if symbol name is not found.
   * @param name symbol name.
   * @return pointer to object, or nullptr.
   */
  const T *get(const std::string &name) const;

  /*!
   * Check whether symbol name exists in the current scope.
   * @param name symbol name.
   * @return whether symbol name exists.
   */
  bool contains(const std::string &name) const;

  const std::string &getIdentifier() const { return identifier; }

private:
  /*!
   * Identifier of the current scope, used for debugging and safety check.
   */
  const std::string identifier;

  /*!
   * A mapping between symbol name and object.
   */
  std::unordered_map<std::string, T> nameToObject;
};

/*!
 * A data structure for representing symbol table.
 */
template <typename T>
struct SymbolMapping {
  SymbolMapping() = default;

  /*!
   *  Get pointer to object by onnx name, or nullptr if onnx name is not found.
   *  @param name onnx name.
   *  @return pointer to object corresponding to `name`, or nullptr.
   */
  const T *GetByOnnxName(const std::string &name) const;

  /*!
   *  Add a new mapping from onnx name to object.
   *  @param name onnx name.
   *  @param tensor object.
   */
  void AddMapping(const std::string &name, T object);

  /*!
   * Check whether an object with the specified onnx name exists.
   * @param name onnx name.
   * @return whether an object with the sepcified onnx name exists.
   */
  bool ContainsKey(const std::string &name) const;

  /*!
   * Push a new variable scope with a specified identifier to symbol table.
   * @param scopeIdentifier identifier for the variable scope.
   */
  void pushScope(const std::string &scopeIdentifier);

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
const T *SymbolMapping<T>::GetByOnnxName(const std::string &name) const {
  for (const auto &scope : scopes)
    if (const T *objPtr = scope.get(name))
      return objPtr;
  return nullptr;
}

template <typename T>
void SymbolMapping<T>::AddMapping(const std::string &name, T object) {
  assert(!scopes.empty());
  assert(!scopes.back().contains(name) && "Object already exists.");
  scopes.back().set(name, object);
}

template <typename T>
bool SymbolMapping<T>::ContainsKey(const std::string &name) const {
  return llvm::any_of(scopes,
      [name](const VariableScope<T> &scope) { return scope.contains(name); });
}

template <typename T>
void SymbolMapping<T>::pushScope(const std::string &scopeIdentifier) {
  scopes.emplace_back(VariableScope<T>(scopeIdentifier));
}

template <typename T>
void SymbolMapping<T>::popScope(const std::string &scopeIdentifier) {
  assert(scopes.back().getIdentifier() == scopeIdentifier);
  scopes.pop_back();
}

template <typename T>
void VariableScope<T>::set(const std::string &name, T object) {
  assert(nameToObject.count(name) == 0 && "duplicate key in symbol table");
  nameToObject.emplace(name, object);
}

template <typename T>
const T *VariableScope<T>::get(const std::string &name) const {
  auto iter = nameToObject.find(name);
  return iter == nameToObject.end() ? nullptr : &(iter->second);
}

template <typename T>
bool VariableScope<T>::contains(const std::string &name) const {
  return nameToObject.count(name) > 0;
}

} // namespace onnx_mlir
#endif
