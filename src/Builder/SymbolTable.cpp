#include "SymbolTable.hpp"

namespace onnx_mlir {

mlir::Value SymbolMapping::GetTensorByOnnxName(const std::string &name) {
  for (const auto &scope : _scopes)
    if (scope.contain(name))
      return scope.get(name);
  llvm_unreachable("Tensor not found");
}

void SymbolMapping::AddMapping(const std::string &name, mlir::Value tensor) {
  assert(!_scopes.empty());
  assert(!_scopes.back().contain(name) && "Tensor already exists.");
  _scopes.back().set(name, tensor);
}

bool SymbolMapping::ContainKey(const std::string &name) {
  return llvm::any_of(_scopes,
      [name](const VariableScope &scope) { return scope.contain(name); });
}

void SymbolMapping::pushScope(const std::string &identifier) {
  _scopes.emplace_back(VariableScope(identifier));
}

void SymbolMapping::popScope(const std::string &scopeIdentifier) {
  assert(_scopes.back().identifier == scopeIdentifier);
  _scopes.pop_back();
}

void VariableScope::set(const std::string &name, mlir::Value val) {
  assert(_nameToValue.count(name) == 0 && "duplicate key in symbol table");
  _nameToValue.emplace(name, val);
}

mlir::Value VariableScope::get(const std::string &name) const {
  return _nameToValue.at(name);
}

bool VariableScope::contain(const std::string &name) const {
  return _nameToValue.count(name) > 0;
}

} // namespace onnx_mlir