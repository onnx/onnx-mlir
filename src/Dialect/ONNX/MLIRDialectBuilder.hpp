//===- MLIRDialectBuilder.hpp - Support migration to MLIR without EDSC --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef ONNX_AND_MLIR_DIALECT_BUILDER_H
#define ONNX_AND_MLIR_DIALECT_BUILDER_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {

struct DialectBuilder {
  DialectBuilder(OpBuilder &b, Location loc) : b(b), loc(loc) {}
  DialectBuilder(ImplicitLocOpBuilder &lb) : b(lb), loc(lb.getLoc()) {}
  DialectBuilder(DialectBuilder &db) : b(db.b), loc(db.loc) {}

  OpBuilder &getBuilder() { return b; }
  Location getLoc() { return loc; }

protected:
  OpBuilder &b;
  Location loc;
};

//===----------------------------------------------------------------------===//
// Adapted from mlir/Dialect/StandardOps/Utils/Utils.h
//===----------------------------------------------------------------------===//

/// Helper struct to build simple arithmetic quantities with minimal type
/// inference support. Code is adapted to support the DialectBuilder super-class
/// that facilitate the building of other dialect builders using another dialect
/// builder.

struct MathBuilder : DialectBuilder {
  MathBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  MathBuilder(ImplicitLocOpBuilder &lb) : DialectBuilder(lb) {}
  MathBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value _and(Value lhs, Value rhs);
  Value add(Value lhs, Value rhs);
  Value sub(Value lhs, Value rhs);
  Value mul(Value lhs, Value rhs);
  Value select(Value cmp, Value lhs, Value rhs);
  Value sgt(Value lhs, Value rhs);
  Value slt(Value lhs, Value rhs);
};

} // namespace mlir
#endif