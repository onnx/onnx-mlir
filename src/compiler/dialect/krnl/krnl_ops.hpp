//===--------------------- krnl_ops.hpp - MLIR Operations -----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "src/compiler/dialect/krnl/krnl_types.hpp"

namespace mlir {
class KrnlOpsDialect : public Dialect {
 public:
  KrnlOpsDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "krnl"; }

//  /// Parse a type registered to this dialect. Overriding this method is
//  /// required for dialects that have custom types.
//  /// Technically this is only needed to be able to round-trip to textual IR.
//  mlir::Type parseType(
//      llvm::StringRef tyData, mlir::Location loc) const override {
//    MLIRContext* context = getContext();
//
//    if (tyData.consume_front("loop"))
//      return LoopType::get(context);
//    else
//      return (emitError(loc, "Unexpected type: " + tyData), Type());
//  }
//
//  /// Print a type registered to this dialect. Overriding this method is
//  /// only required for dialects that have custom types.
//  /// Technically this is only needed to be able to round-trip to textual IR.
//  void printType(mlir::Type type, llvm::raw_ostream& os) const override {
//    switch (type.getKind()) {
//      case KrnlTypes::Loop:
//        os << "loop";
//        return;
//    }
//  }
};

#define GET_OP_CLASSES
#include "src/compiler/krnl.hpp.inc"
}  // namespace mlir
