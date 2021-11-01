//===---- MLIRDialectBuilder.hpp - Helper functions for MLIR dialects -----===//
//
// Copyright 2019-2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for building MLIR operations.
//
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Original code for MathBuilder is copied from LLVM MLIR Utils.cpp
// Modified here to add operations, add super class.
// Liscense added here for this class for completness.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

struct MathBuilder final : DialectBuilder {
  MathBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  MathBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value _and(Value lhs, Value rhs) const;
  Value _or(Value lhs, Value rhs) const;

  Value add(Value lhs, Value rhs) const;
  Value sub(Value lhs, Value rhs) const;
  Value mul(Value lhs, Value rhs) const;
  Value div(Value lhs, Value rhs) const;
  Value exp(Value val) const;
  Value exp2(Value val) const;
  Value log2(Value val) const;

  Value select(Value cmp, Value lhs, Value rhs) const;
  Value sgt(Value lhs, Value rhs) const;
  Value sge(Value lhs, Value rhs) const;
  Value slt(Value lhs, Value rhs) const;
  Value sle(Value lhs, Value rhs) const;
  Value eq(Value lhs, Value rhs) const;
  Value neq(Value lhs, Value rhs) const;
  Value min(Value lhs, Value rhs) const;
  Value max(Value lhs, Value rhs) const;

  Value constant(Type type, double val) const;
  Value constantIndex(int64_t val) const;

  // Cast handle bool/int/float/index elementary types. Do not convert
  // signed/index to unsigned.
  Value cast(Type destType, Value val) const;
  Value castToIndex(Value val) const;

private:
  Value castToSignless(Value source, int64_t width) const;
  Value castToUnsigned(Value source, int64_t width) const;
};

//===----------------------------------------------------------------------===//
// MemRef Builder with added support for aligned memory
//===----------------------------------------------------------------------===//

struct MemRefBuilder final : DialectBuilder {
  MemRefBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  MemRefBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  memref::AllocOp alloc(MemRefType type) const;
  memref::AllocOp alloc(MemRefType type, ValueRange dynSymbols) const;
  memref::AllocOp alignedAlloc(MemRefType type, int64_t align = -1) const;
  memref::AllocOp alignedAlloc(
      MemRefType type, ValueRange dynSymbols, int64_t align = -1) const;

  memref::AllocaOp alloca(MemRefType type) const;
  memref::AllocaOp alignedAlloca(MemRefType type, int64_t align = -1) const;

  memref::DeallocOp dealloc(Value val) const;

  memref::CastOp cast(Value input, MemRefType outputType) const;

  Value dim(Value val, int64_t index) const;
};

// Default alignment attribute for all allocation of memory. On most system, it
// is 16 bytes.
static constexpr int64_t gDefaultAllocAlign = 16;

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF) Builder
//===----------------------------------------------------------------------===//

struct SCFBuilder final : DialectBuilder {
  SCFBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  SCFBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  /// Create an if then with optional else. Construct does not generate a result
  /// (unlike some scf::if) and introduces the yields automatically.
  void ifThenElse(Value cond, function_ref<void(SCFBuilder &createSCF)> thenFn,
      function_ref<void(SCFBuilder &createSCF)> elseFn = nullptr) const;

  void yield() const;
};

//===----------------------------------------------------------------------===//
// Affine Builder
//===----------------------------------------------------------------------===//

template <class LOAD_OP, class STORE_OP>
struct GenericAffineBuilder final : DialectBuilder {
  GenericAffineBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  GenericAffineBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value load(Value memref, ValueRange indices = {}) const;
  void store(Value val, Value memref, ValueRange indices = {}) const;

  void forIE(IndexExpr lb, IndexExpr ub, int64_t step,
      function_ref<void(GenericAffineBuilder &, Value)> builderFn) const;

  void forIE(SmallVectorImpl<IndexExpr> &lbs, SmallVectorImpl<IndexExpr> &ubs,
      SmallVectorImpl<int64_t> &steps,
      function_ref<void(GenericAffineBuilder &, ValueRange)> builderFn) const;

  // This if then else construct has no arguments to the blocks.
  void ifThenElse(IndexExprScope &scope, SmallVectorImpl<IndexExpr> &conditions,
      function_ref<void(GenericAffineBuilder &createAffine)> thenFn,
      function_ref<void(GenericAffineBuilder &createAffine)> elseFn) const;

  void yield() const;

private:
  // Support for multiple forIE loops.
  void recursionForIE(SmallVectorImpl<IndexExpr> &lbs,
      SmallVectorImpl<IndexExpr> &ubs, SmallVectorImpl<int64_t> &steps,
      SmallVectorImpl<Value> &loopIndices,
      function_ref<void(GenericAffineBuilder &, ValueRange)> builderFn) const;

  // Support for adding blocks.
  void appendToBlock(
      Block *block, function_ref<void(ValueRange)> builderFn) const;
};

// Include template implementations.
#include "MLIRDialectBuilder.hpp.inc"

} // namespace mlir
#endif
