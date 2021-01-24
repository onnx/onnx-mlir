#include "SymbolTable.hpp"

namespace onnx_mlir {
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