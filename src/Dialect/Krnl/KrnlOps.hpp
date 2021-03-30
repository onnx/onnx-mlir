/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- KrnlOps.hpp - Krnl Operations ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of krnl operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "src/Interface/SpecializedKernelOpInterface.hpp"
#include "llvm/ADT/TypeSwitch.h"

#include "KrnlHelper.hpp"
#include "KrnlTypes.hpp"

namespace mlir {
class KrnlOpsDialect : public Dialect {
public:
  KrnlOpsDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "krnl"; }

  /// Parse a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override {
    if (succeeded(parser.parseOptionalKeyword("loop")))
      return LoopType::get(parser.getBuilder().getContext());

    parser.emitError(parser.getCurrentLocation(), "Unknown type");
    return {};
  }

  /// Print a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override {
    TypeSwitch<Type>(type).Case<LoopType>([&](Type) {
      os << "loop";
      return;
    });
  }
};
} // namespace mlir

#define GET_OP_CLASSES
#include "src/Dialect/Krnl/KrnlOps.hpp.inc"
