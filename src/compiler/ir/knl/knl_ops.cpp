#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "knl_ops.hpp"

namespace mlir {
KnlOpsDialect::KnlOpsDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "src/compiler/knl.cpp.inc"
      >();
}
}  // namespace mlir

namespace onnf {}
