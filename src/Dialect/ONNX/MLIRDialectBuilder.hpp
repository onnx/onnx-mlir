//===- MLIRDialectBuilder.hpp - Support migration to MLIR without EDSC --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef ONNX_AND_MLIR_DIALECT_BUILDER_H
#define ONNX_AND_MLIR_DIALECT_BUILDER_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
  Value div(Value lhs, Value rhs);
  Value exp(Value val);
  Value select(Value cmp, Value lhs, Value rhs);
  Value sgt(Value lhs, Value rhs);
  Value slt(Value lhs, Value rhs);
};

struct MemRefBuilder : DialectBuilder {
  MemRefBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  MemRefBuilder(ImplicitLocOpBuilder &lb) : DialectBuilder(lb) {}
  MemRefBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  // Alloc.
  memref::AllocOp alloc(MemRefType type);
  memref::AllocOp alloc(MemRefType type, ValueRange dynSymbols);
  memref::AllocOp alignedAlloc(MemRefType type, int64_t align = -1);
  memref::AllocOp alignedAlloc(
      MemRefType type, ValueRange dynSymbols, int64_t align = -1);
  // Alloca.
  memref::AllocaOp alloca(MemRefType type);
  memref::AllocaOp alignedAlloca(MemRefType type, int64_t align = -1);
  // Dealloc.
  memref::DeallocOp dealloc(Value val);
  // DimOp
  Value dim(Value val, int64_t index);
};

// Default alignment attribute for all allocation of memory. On most system, it
// is 16 bytes.
// TODO: make it a global variable
// extern int64_t gDefaultAllocAlign;
#define gDefaultAllocAlign 16

} // namespace mlir
#endif
