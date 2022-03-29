//===---- MLIRDialectBuilder.hpp - Helper functions for MLIR dialects -----===//
//
// Copyright 2019-2022 The IBM Research Authors.
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

  Value andi(Value lhs, Value rhs) const;
  Value ori(Value lhs, Value rhs) const;

  Value add(Value lhs, Value rhs) const;
  Value sub(Value lhs, Value rhs) const;
  Value mul(Value lhs, Value rhs) const;
  Value div(Value lhs, Value rhs) const;
  Value exp(Value val) const;
  Value exp2(Value val) const;
  Value log2(Value val) const;
  Value sqrt(Value val) const;
  Value pow(Value base, Value exp) const;

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

  /// Emit a negative infinity constant of a specific type. Supported types:
  /// F16, F32, F64, Int8, Int16, Int32, Int64. In case of Float, emit the
  /// negative of the positive infinity. In case of Integer, emit the minimum
  /// value.
  Value negativeInf(Type type) const;

  /// Emit a positive infinity constant of a specific type. Supported types:
  /// F16, F32, F64, Int8, Int16, Int32, Int64. In case of Integer, emit the
  /// maximum value.
  Value positiveInf(Type type) const;

  // Cast handle bool/int/float/index elementary types. Do not convert
  // signed/index to unsigned.
  Value cast(Type destType, Value val) const;
  Value castToIndex(Value val) const;

private:
  Value createArithCmp(Value lhs, Value rhs, arith::CmpIPredicate pred) const;
  Value createArithCmp(Value lhs, Value rhs, arith::CmpFPredicate pred) const;
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

  Value reinterpretCast(
      Value input, SmallVectorImpl<IndexExpr> &outputDims) const;
  Value dim(Value val, int64_t index) const;
  Value dim(Value val, Value index) const;
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
// Vector Builder
//===----------------------------------------------------------------------===//

struct VectorBuilder final : DialectBuilder {
  VectorBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  VectorBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value load(VectorType vecType, Value memref, ValueRange indices = {}) const;
  void store(Value val, Value memref, ValueRange indices = {}) const;

  Value broadcast(VectorType vecType, Value val) const;
  Value shuffle(Value lhs, Value rhs, SmallVectorImpl<int64_t> &mask) const;
  Value fma(Value lhs, Value rhs, Value acc) const;

  // Composite functions
  Value mergeLow(Value lhs, Value rhs, int64_t step);
  Value mergeHigh(Value lhs, Value rhs, int64_t step);
  Value multiReduction(SmallVectorImpl<Value> &vecArray); // Only 4x4 as of now.

private:
  bool isPowerOf2(uint64_t num);
  uint64_t vector1DLength(Value vec);
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

// Affine builder uses affine load and store for memory operations. A later
// definition of AffineBuilderKrnlMem will use Krnl load and store for memory
// operations. We recommend to use AffineBuilderKrnlMem when converting the Krnl
// dialect into the affine dialect.
using AffineBuilder = GenericAffineBuilder<AffineLoadOp, AffineStoreOp>;

//===----------------------------------------------------------------------===//
// Multi Dialect Builder
//===----------------------------------------------------------------------===//

/*
  Instead of creating multiple builders, e.g.

  KrnlBuilder createKrnl(rewriter, loc);
  MathBuilder createMath(createKrnl);
  MemRefBuilder createMemRef(createKrnl);

  createKrnl.defineLoop(1);
  createMath.add(i1, i2);
  createMemRef.alloca(type);

  We can create a single builder composed of multiple types

  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder>
    create(rewriter, loc);

  create.krnl.defineLoop(1);
  create.math.add(i1, i2);
  create.mem.alloca(type);

  Types that can be used here are
  *  AffineBuilder, access field with affine
  *  AffineBuilderKrnlMem, access field with affineKMem
  *  KrnlBuilder, access field with krnl
  *  MathBuilder, access field with math
  *  MemRefBuilder, access field with mem
  *  ONNXBuilder, access field with onnx
  *  SCFBuilder, access field with scf

*/

// Anchor class.
template <class... Ts>
struct MultiDialectBuilder {
  MultiDialectBuilder(OpBuilder &b, Location loc) {}
  MultiDialectBuilder(DialectBuilder &db) {}
};

// Recursive class specialized for MathBuilder refereed to as math.
template <class... Ts>
struct MultiDialectBuilder<MathBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), math(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), math(db) {}
  MathBuilder math;
};

// Recursive class specialized for MemRefBuilder refereed to as mem.
template <class... Ts>
struct MultiDialectBuilder<MemRefBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), mem(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), mem(db) {}
  MemRefBuilder mem;
};

// Recursive class specialized for AffineBuilder refereed to as affine.
template <class... Ts>
struct MultiDialectBuilder<AffineBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), affine(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), affine(db) {}
  AffineBuilder affine;
};

// Recursive class specialized for SCFBuilder refereed to as scf.
template <class... Ts>
struct MultiDialectBuilder<SCFBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), scf(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), scf(db) {}
  SCFBuilder scf;
};

// Recursive class specialized for VectorBuilder refereed to as scf.
template <class... Ts>
struct MultiDialectBuilder<VectorBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), vec(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), vec(db) {}
  VectorBuilder vec;
};

// Include template implementations.
#include "MLIRDialectBuilder.hpp.inc"

} // namespace mlir
#endif
