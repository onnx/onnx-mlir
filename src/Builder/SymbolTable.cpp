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

std::string legalize_name(std::string name) {
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '-', '_');
  replaceAll(name, ":", "_colon_");
  // If tensor name starts with a number, prepend n to make it a legal c++
  // identifier.
  if (name.size() > 0 && isdigit(name.at(0)))
    name.insert(0, 1, 'n');
  return name;
}

mlir::Value SymbolMapping::GetTensorByOnnxName(const std::string &name) {
  std::string legalized_name = legalize_name(name);
  for (const auto &scope : _scopes)
    if (scope.contain(legalized_name))
      return scope.get(legalized_name);
  llvm_unreachable("Tensor not found");
}

void SymbolMapping::AddMapping(const std::string &name, mlir::Value tensor) {
  assert(!_scopes.empty());
  assert(
      !_scopes.back().contain(legalize_name(name)) && "Tensor already exists.");
  _scopes.back().set(legalize_name(name), tensor);
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
  _nameToValue.emplace(name, val);
}

mlir::Value VariableScope::get(std::string name) const {
  return _nameToValue.at(name);
}

bool VariableScope::contain(std::string name) const {
  return _nameToValue.count(name) > 0;
}

} // namespace onnx_mlir