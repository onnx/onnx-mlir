#pragma once

#include <map>
#include <string>
#include <vector>

#include "mlir/IR/Value.h"
#include "onnx/onnx_pb.h"

namespace onnx_mlir {

struct VariableScope {
  explicit VariableScope(std::string identifier)
      : identifier(std::move(identifier)){};

  void set(std::string name, mlir::Value value);

  mlir::Value get(std::string name) const;

  bool contain(std::string name) const;

  const std::string identifier;

private:
  std::map<std::string, mlir::Value> _nameToValue;

  std::map<std::string, onnx::ValueInfoProto> _nameToValueInfo;
};

} // namespace onnx_mlir
