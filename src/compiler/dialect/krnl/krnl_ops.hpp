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
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "src/compiler/dialect/krnl/krnl_types.hpp"

namespace mlir {
class KrnlOpsDialect : public Dialect {
 public:
  KrnlOpsDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "krnl"; }
  
  /// Parse a type registered to this dialect.
  Type parseType(DialectAsmParser& parser) const override {
    if (succeeded(parser.parseOptionalKeyword("loop")))
      return LoopType::get(parser.getBuilder().getContext());

    parser.emitError(parser.getCurrentLocation(), "Unknown type");
  }

  /// Print a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter& os) const override {
    switch (type.getKind()) {
      case KrnlTypes::Loop:
        os << "loop";
        return;
    }
  }
>>>>>>> 011cc1b... Update type parser (#376)
};

#define GET_OP_CLASSES
#include "src/compiler/krnl.hpp.inc"
}  // namespace mlir
