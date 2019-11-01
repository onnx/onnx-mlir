#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
class KnlOpsDialect : public Dialect {
 public:
  KnlOpsDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "knl"; }
};

#define GET_OP_CLASSES
#include "src/compiler/knl.hpp.inc"
}  // namespace mlir

namespace onnf {}
