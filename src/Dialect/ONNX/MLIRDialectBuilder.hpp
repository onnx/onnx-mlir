//===- MLIRDialectBuilder.hpp - Support migration to MLIR without EDSC --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef ONNX_AND_MLIR_DIALECT_BUILDER_H
#define ONNX_AND_MLIR_DIALECT_BUILDER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include "IndexExpr.hpp"

namespace mlir {

struct DialectBuilder {
  DialectBuilder(OpBuilder &b, Location loc) : b(b), loc(loc) {}
  DialectBuilder(DialectBuilder &db) : b(db.b), loc(db.loc) {}
  virtual ~DialectBuilder() {}
  DialectBuilder(DialectBuilder &&) = delete;
  DialectBuilder &operator=(const DialectBuilder &) = delete;
  DialectBuilder &&operator=(const DialectBuilder &&) = delete;

  OpBuilder &getBuilder() const { return b; }
  Location getLoc() const { return loc; }

protected:
  OpBuilder &b;
  Location loc;
};

//===----------------------------------------------------------------------===//
// Math Builder
//===----------------------------------------------------------------------===//

/// Helper struct to build simple arithmetic quantities with minimal type
/// inference support. Code is adapted to support the DialectBuilder super-class
/// that facilitate the building of other dialect builders using another dialect
/// builder.

// Adapted from mlir/Dialect/StandardOps/Utils/Utils.h

struct MathBuilder : DialectBuilder {
  MathBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  MathBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value _and(Value lhs, Value rhs);
  Value _or(Value lhs, Value rhs);

  Value add(Value lhs, Value rhs);
  Value sub(Value lhs, Value rhs);
  Value mul(Value lhs, Value rhs);
  Value div(Value lhs, Value rhs);
  Value exp(Value val);
  Value exp2(Value val);
  Value log2(Value val);

  Value select(Value cmp, Value lhs, Value rhs);
  Value sgt(Value lhs, Value rhs);
  Value sge(Value lhs, Value rhs);
  Value slt(Value lhs, Value rhs);
  Value eq(Value lhs, Value rhs);

  Value constant(Type type, double val);
  Value constantIndex(int64_t val);
};

//===----------------------------------------------------------------------===//
// MemRef Builder with added support for aligned memory
//===----------------------------------------------------------------------===//

struct MemRefBuilder : DialectBuilder {
  MemRefBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
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
static constexpr int64_t gDefaultAllocAlign = 16;

//===----------------------------------------------------------------------===//
// Affine Builder
//===----------------------------------------------------------------------===//

template <class LOAD_OP, class STORE_OP>
struct GenericAffineBuilder : DialectBuilder {
  GenericAffineBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  GenericAffineBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value load(Value memref, ValueRange indices = {});
  void store(Value val, Value memref, ValueRange indices = {});

  void forIE(IndexExpr lb, IndexExpr ub, int64_t step,
      function_ref<void(GenericAffineBuilder &, Value)> builderFn);

  void forIE(SmallVectorImpl<IndexExpr> &lbs, SmallVectorImpl<IndexExpr> &ubs,
      SmallVectorImpl<int64_t> &steps,
      function_ref<void(GenericAffineBuilder &, ValueRange)> builderFn);

  // This if then else construct has no arguments to the blocks.
  void ifThenElse(IndexExprScope &scope, SmallVectorImpl<IndexExpr> &conditions,
      function_ref<void(GenericAffineBuilder &createAffine)> thenFn,
      function_ref<void(GenericAffineBuilder &createAffine)> elseFn);

  void yield();

private:
  // Support for multiple forIE loops.
  void recursionForIE(SmallVectorImpl<IndexExpr> &lbs,
      SmallVectorImpl<IndexExpr> &ubs, SmallVectorImpl<int64_t> &steps,
      SmallVectorImpl<Value> &loopIndices,
      function_ref<void(GenericAffineBuilder &, ValueRange)> builderFn);

  // Support for adding blocks.
  void appendToBlock(Block *block, function_ref<void(ValueRange)> builderFn);
};

// Include template implementations.
#include "MLIRDialectBuilder.hpp.inc"

} // namespace mlir
#endif
