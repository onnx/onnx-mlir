template<typename T>
T SymbolMapping<T>::GetTensorByOnnxName(const std::string &name) {
  for (const auto &scope : _scopes)
    if (scope.contain(name))
      return scope.get(name);
  llvm_unreachable("Tensor not found");
}

template<typename T>
void SymbolMapping<T>::AddMapping(const std::string &name, T tensor) {
  assert(!_scopes.empty());
  assert(!_scopes.back().contain(name) && "Tensor already exists.");
  _scopes.back().set(name, tensor);
}

template<typename T>
bool SymbolMapping<T>::ContainKey(const std::string &name) {
  return llvm::any_of(_scopes,
      [name](const VariableScope<T> &scope) { return scope.contain(name); });
}

template<typename T>
void SymbolMapping<T>::pushScope(const std::string &identifier) {
  _scopes.emplace_back(VariableScope<T>(identifier));
}

template<typename T>
void SymbolMapping<T>::popScope(const std::string &scopeIdentifier) {
  assert(_scopes.back().identifier == scopeIdentifier);
  _scopes.pop_back();
}

template<typename T>
void VariableScope<T>::set(const std::string &name, T val) {
  assert(_nameToValue.count(name) == 0 && "duplicate key in symbol table");
  _nameToValue.emplace(name, val);
}

template<typename T>
T VariableScope<T>::get(const std::string &name) const {
  return _nameToValue.at(name);
}

template<typename T>
bool VariableScope<T>::contain(const std::string &name) const {
  return _nameToValue.count(name) > 0;
}
