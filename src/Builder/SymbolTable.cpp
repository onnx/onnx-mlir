#include "SymbolTable.hpp"

namespace onnx_mlir {

void replaceAll(
    std::string &str, const std::string &from, const std::string &to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // In case 'to' contains 'from', like replacing
                              // 'x' with 'yx'
  }
}

mlir::Value SymbolMapping::GetTensorByOnnxName(const std::string &name) {
  std::string result;
  result = name;
  std::string legalized_name = result;
  for (const auto &scope : _scopes)
    if (scope.contain(legalized_name))
      return scope.get(legalized_name);
  llvm_unreachable("Tensor not found");
}

void SymbolMapping::AddMapping(const std::string &name, mlir::Value tensor) {
  assert(!_scopes.empty());
  assert({
    std::string name1 = name;
    std::string result1;
    result1 = name1;
    !_scopes.back().contain(result1) && "Tensor already exists."
  });
  std::string result;
  result = name;
  _scopes.back().set(result, tensor);
}

bool SymbolMapping::ContainKey(std::string name) {
  return llvm::any_of(
      _scopes, [name](VariableScope scope) { return scope.contain(name); });
}

void SymbolMapping::pushScope(const std::string &identifier) {
  _scopes.emplace_back(VariableScope(identifier));
}

void SymbolMapping::popScope(const std::string &scopeIdentifier) {
  assert(_scopes.back().identifier == scopeIdentifier);
  _scopes.pop_back();
}

void VariableScope::set(std::string name, mlir::Value val) {
  assert(_nameToValue.count(name) == 0 && "duplicate key in symbol table");
  _nameToValue.emplace(name, val);
}

mlir::Value VariableScope::get(std::string name) const {
  return _nameToValue.at(name);
}

bool VariableScope::contain(std::string name) const {
  return _nameToValue.count(name) > 0;
}

} // namespace onnx_mlir