//===---- DialectBuilder.hpp - Helper functions for MLIR dialects -----===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for building MLIR operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"

namespace onnx_mlir {

struct DialectBuilder {
  DialectBuilder(mlir::OpBuilder &b, mlir::Location loc) : b(b), loc(loc) {}
  DialectBuilder(const DialectBuilder &db) : b(db.b), loc(db.loc) {}
  virtual ~DialectBuilder() {}
  DialectBuilder(DialectBuilder &&) = delete;
  DialectBuilder &operator=(const DialectBuilder &) = delete;
  DialectBuilder &&operator=(const DialectBuilder &&) = delete;

  mlir::OpBuilder &getBuilder() const { return b; }
  mlir::Location getLoc() const { return loc; }

protected:
  mlir::OpBuilder &b;
  mlir::Location loc;
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
// License added here for this class for completeness.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

struct MathBuilder final : DialectBuilder {
  MathBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  MathBuilder(const DialectBuilder &db) : DialectBuilder(db) {}

  mlir::Value abs(mlir::Value val) const;

  mlir::Value andi(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value ori(mlir::Value lhs, mlir::Value rhs) const;

  mlir::Value add(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value sub(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value mul(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value div(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value exp(mlir::Value val) const;
  mlir::Value exp2(mlir::Value val) const;
  mlir::Value log2(mlir::Value val) const;
  mlir::Value sqrt(mlir::Value val) const;
  mlir::Value pow(mlir::Value base, mlir::Value exp) const;

  mlir::Value select(mlir::Value cmp, mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value sgt(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value sge(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value slt(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value sle(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value eq(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value neq(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value min(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value max(mlir::Value lhs, mlir::Value rhs) const;

  mlir::Value constant(mlir::Type type, double val) const;
  mlir::Value constantIndex(int64_t val) const;

  /// Emit a negative infinity constant of a specific type. Supported types:
  /// F16, F32, F64, Int8, Int16, Int32, Int64. In case of Float, emit the
  /// negative of the positive infinity. In case of Integer, emit the minimum
  /// mlir::Value.
  mlir::Value negativeInf(mlir::Type type) const;

  /// Emit a positive infinity constant of a specific type. Supported types:
  /// F16, F32, F64, Int8, Int16, Int32, Int64. In case of Integer, emit the
  /// maximum mlir::Value.
  mlir::Value positiveInf(mlir::Type type) const;

  // Cast handle bool/int/float/index elementary types. Do not convert
  // signed/index to unsigned.
  mlir::Value cast(mlir::Type destType, mlir::Value val) const;
  mlir::Value castToIndex(mlir::Value val) const;

  // Add indexOffsets to the least significant indices. So if indices are (i, j,
  // k, l) and offsets are (K, L), the results will be (i, j, k+K, l+L).
  void addOffsetToLeastSignificant(mlir::ValueRange indices,
      mlir::ValueRange offsets,
      llvm::SmallVectorImpl<mlir::Value> &computedIndices) const;
  void addOffsetToLeastSignificant(mlir::ArrayRef<IndexExpr> indices,
      mlir::ValueRange offsets,
      llvm::SmallVectorImpl<mlir::Value> &computedIndices) const;

private:
  mlir::Value createArithCmp(
      mlir::Value lhs, mlir::Value rhs, mlir::arith::CmpIPredicate pred) const;
  mlir::Value createArithCmp(
      mlir::Value lhs, mlir::Value rhs, mlir::arith::CmpFPredicate pred) const;
  mlir::Value castToSignless(mlir::Value source, int64_t width) const;
  mlir::Value castToUnsigned(mlir::Value source, int64_t width) const;
};

//===----------------------------------------------------------------------===//
// MemRef Builder with added support for aligned memory
//===----------------------------------------------------------------------===//

struct MemRefBuilder final : DialectBuilder {
  MemRefBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  MemRefBuilder(const DialectBuilder &db) : DialectBuilder(db) {}

  mlir::memref::AllocOp alloc(mlir::MemRefType type) const;
  mlir::memref::AllocOp alloc(
      mlir::MemRefType type, mlir::ValueRange dynSymbols) const;
  mlir::memref::AllocOp alignedAlloc(
      mlir::MemRefType type, int64_t align = -1) const;
  mlir::memref::AllocOp alignedAlloc(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t align = -1) const;

  // The alloca instruction allocates memory on the stack frame of the currently
  // executing function, to be automatically released when this function returns
  // to its caller. It is strongly suggested to place alloca instructions
  // outside of a loop.
  mlir::memref::AllocaOp alloca(mlir::MemRefType type) const;
  mlir::memref::AllocaOp alignedAlloca(
      mlir::MemRefType type, int64_t align = -1) const;

  mlir::memref::DeallocOp dealloc(mlir::Value val) const;

  mlir::memref::CastOp cast(
      mlir::Value input, mlir::MemRefType outputType) const;

  mlir::Value reinterpretCast(
      mlir::Value input, llvm::SmallVectorImpl<IndexExpr> &outputDims) const;
  mlir::Value dim(mlir::Value val, int64_t index) const;
  mlir::Value dim(mlir::Value val, mlir::Value index) const;
};

// Default alignment attribute for all allocation of memory. On most system, it
// is 16 bytes.
static constexpr int64_t gDefaultAllocAlign = 16;

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF) Builder
//===----------------------------------------------------------------------===//

struct SCFBuilder final : DialectBuilder {
  SCFBuilder(mlir::OpBuilder &b, mlir::Location loc) : DialectBuilder(b, loc) {}
  SCFBuilder(const DialectBuilder &db) : DialectBuilder(db) {}

  /// Create an if then with optional else. Construct does not generate a result
  /// (unlike some scf::if) and introduces the yields automatically.
  void ifThenElse(mlir::Value cond,
      mlir::function_ref<void(SCFBuilder &createSCF)> thenFn,
      mlir::function_ref<void(SCFBuilder &createSCF)> elseFn = nullptr) const;

  void parallelLoop(mlir::ValueRange lowerBounds, mlir::ValueRange upperBounds,
      mlir::ValueRange steps,
      mlir::function_ref<void(DialectBuilder &, mlir::ValueRange)> bodyFn)
      const;
  void yield() const;
};

//===----------------------------------------------------------------------===//
// Vector Builder
//===----------------------------------------------------------------------===//

struct VectorBuilder final : DialectBuilder {
  VectorBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  VectorBuilder(const DialectBuilder &db) : DialectBuilder(db) {}

  // Get the machine SIMD vector length for the given elementary type.
  // This can help guide certain optimizations.
  int64_t getMachineVectorLength(const mlir::Type &elementType) const;
  int64_t getMachineVectorLength(const mlir::VectorType &vecType) const;
  int64_t getMachineVectorLength(mlir::Value vecValue) const;

  mlir::Value load(mlir::VectorType vecType, mlir::Value memref,
      mlir::ValueRange indices = {}) const;
  // When ranks of offsets<indices, add offsets to the least significant dims.
  mlir::Value load(mlir::VectorType vecType, mlir::Value memref,
      mlir::ValueRange indices, mlir::ValueRange offsets) const;
  mlir::Value loadIE(mlir::VectorType vecType, mlir::Value memref,
      llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) const;
  void store(
      mlir::Value val, mlir::Value memref, mlir::ValueRange indices = {}) const;
  // When ranks of offsets<indices, add offsets to the least significant dims.
  void store(mlir::Value val, mlir::Value memref, mlir::ValueRange indices,
      mlir::ValueRange offsets) const;
  void storeIE(mlir::Value val, mlir::Value memref,
      llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) const;

  mlir::Value broadcast(mlir::VectorType vecType, mlir::Value val) const;
  mlir::Value shuffle(mlir::Value lhs, mlir::Value rhs,
      llvm::SmallVectorImpl<int64_t> &mask) const;
  mlir::Value fma(mlir::Value lhs, mlir::Value rhs, mlir::Value acc) const;

  // Composite functions.
  mlir::Value mergeHigh(mlir::Value lhs, mlir::Value rhs, int64_t step) const;
  mlir::Value mergeLow(mlir::Value lhs, mlir::Value rhs, int64_t step) const;
  void multiReduction(llvm::SmallVectorImpl<mlir::Value> &inputVecArray,
      llvm::SmallVectorImpl<mlir::Value> &outputVecArray);

private:
  bool isPowerOf2(uint64_t num) const;
  uint64_t getLengthOf1DVector(mlir::Value vec) const;
};

//===----------------------------------------------------------------------===//
// Affine Builder
//===----------------------------------------------------------------------===//

template <class LOAD_OP, class STORE_OP>
struct GenericAffineBuilder final : DialectBuilder {
  GenericAffineBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  GenericAffineBuilder(const DialectBuilder &db) : DialectBuilder(db) {}

  mlir::Value load(mlir::Value memref, mlir::ValueRange indices = {}) const;
  // When ranks of offsets<indices, add offsets to the least significant dims.
  mlir::Value load(mlir::Value memref, mlir::ValueRange indices,
      mlir::ValueRange offsets) const;
  mlir::Value loadIE(mlir::Value memref, llvm::ArrayRef<IndexExpr> indices,
      mlir::ValueRange offsets) const;

  void store(
      mlir::Value val, mlir::Value memref, mlir::ValueRange indices = {}) const;
  // When ranks of offsets<indices, add offsets to the least significant dims.
  void store(mlir::Value val, mlir::Value memref, mlir::ValueRange indices,
      mlir::ValueRange offsets) const;
  void storeIE(mlir::Value val, mlir::Value memref,
      llvm::ArrayRef<IndexExpr> indices, mlir::ValueRange offsets) const;

  void forIE(IndexExpr lb, IndexExpr ub, int64_t step,
      mlir::function_ref<void(GenericAffineBuilder &, mlir::Value)> builderFn)
      const;
  void forIE(llvm::SmallVectorImpl<IndexExpr> &lbs,
      llvm::SmallVectorImpl<IndexExpr> &ubs,
      llvm::SmallVectorImpl<int64_t> &steps,
      mlir::function_ref<void(GenericAffineBuilder &, mlir::ValueRange)>
          builderFn) const;

  // This if then else construct has no arguments to the blocks.
  void ifThenElse(IndexExprScope &scope,
      llvm::SmallVectorImpl<IndexExpr> &conditions,
      mlir::function_ref<void(GenericAffineBuilder &createAffine)> thenFn,
      mlir::function_ref<void(GenericAffineBuilder &createAffine)> elseFn)
      const;

  void yield() const;

private:
  // Support for multiple forIE loops.
  void recursionForIE(llvm::SmallVectorImpl<IndexExpr> &lbs,
      llvm::SmallVectorImpl<IndexExpr> &ubs,
      llvm::SmallVectorImpl<int64_t> &steps,
      llvm::SmallVectorImpl<mlir::Value> &loopIndices,
      mlir::function_ref<void(GenericAffineBuilder &, mlir::ValueRange)>
          builderFn) const;

  // Support for adding blocks.
  void appendToBlock(mlir::Block *block,
      mlir::function_ref<void(mlir::ValueRange)> builderFn) const;
};

// Affine builder uses affine load and store for memory operations. A later
// definition of AffineBuilderKrnlMem will use Krnl load and store for memory
// operations. We recommend to use AffineBuilderKrnlMem when converting the Krnl
// dialect into the affine dialect.
using AffineBuilder =
    GenericAffineBuilder<mlir::AffineLoadOp, mlir::AffineStoreOp>;

//===----------------------------------------------------------------------===//
// LLVM Builder
//===----------------------------------------------------------------------===//

struct LLVMBuilder final : DialectBuilder {
  using voidFuncRef = mlir::function_ref<void(LLVMBuilder &createLLVM)>;
  using valueFuncRef = mlir::function_ref<mlir::Value(LLVMBuilder &createLLVM)>;

  LLVMBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  LLVMBuilder(const DialectBuilder &db) : DialectBuilder(db) {}

  // AddressOfOp
  mlir::Value addressOf(mlir::LLVM::GlobalOp op) const;

  // AllocaOp
  mlir::Value _alloca(
      mlir::Type resultType, mlir::Value size, int64_t alignment) const;

  // BitcastOp
  mlir::Value bitcast(mlir::Type type, mlir::Value val) const;
  mlir::Value bitcastI8Ptr(mlir::Value val) const;
  mlir::Value bitcastI8PtrPtr(mlir::Value val) const;

  // BrOp
  void br(
      llvm::ArrayRef<mlir::Value> destOperands, mlir::Block *destBlock) const;

  // CallOp
  mlir::Value call(mlir::ArrayRef<mlir::Type> resultTypes,
      llvm::StringRef funcName, mlir::ArrayRef<mlir::Value> inputs) const;
  mlir::Value call(mlir::ArrayRef<mlir::Type> resultTypes,
      mlir::FlatSymbolRefAttr funcSymbol,
      mlir::ArrayRef<mlir::Value> inputs) const;

  // CondBrOp
  void condBr(mlir::Value cond, mlir::Block *trueBlock,
      llvm::ArrayRef<mlir::Value> trueOperands, mlir::Block *falseBlock,
      llvm::ArrayRef<mlir::Value> falseOperands) const;

  // ConstantOp
  mlir::Value constant(mlir::Type type, int64_t val) const;
  mlir::Value constant(mlir::Type type, double val) const;

  // ExtractValueOp
  mlir::Value extractValue(mlir::Type resultType, mlir::Value container,
      llvm::ArrayRef<int64_t> position) const;

  // FuncOp
  mlir::LLVM::LLVMFuncOp func(llvm::StringRef name, mlir::Type type) const;

  // GEPOp
  mlir::Value getElemPtr(mlir::Type resultType, mlir::Value base,
      llvm::ArrayRef<mlir::Value> indices) const;

  // GlobalOp
  mlir::LLVM::GlobalOp globalOp(mlir::Type resultType, bool isConstant,
      mlir::LLVM::Linkage, llvm::StringRef name, mlir::Attribute attr,
      uint64_t alignment = 0) const;

  // ICmpOp
  mlir::Value icmp(
      mlir::LLVM::ICmpPredicate cond, mlir::Value lhs, mlir::Value rhs) const;

  // InsertValueOp
  mlir::Value insertValue(mlir::Type resultType, mlir::Value container,
      mlir::Value val, llvm::ArrayRef<int64_t> position) const;

  // LoadOp
  mlir::Value load(mlir::Value addr) const;

  // NullOp
  mlir::Value null(mlir::Type type) const;
  mlir::Value nullI8Ptr() const;

  // ReturnOp
  void _return(mlir::Value val) const;

  // StoreOp
  void store(mlir::Value val, mlir::Value addr) const;

  //===--------------------------------------------------------------------===//
  // Helper functions
  //===--------------------------------------------------------------------===//

  // Get or insert a function declaration at the beginning of the module.
  mlir::FlatSymbolRefAttr getOrInsertSymbolRef(mlir::ModuleOp module,
      llvm::StringRef symName, mlir::Type resultType,
      llvm::ArrayRef<mlir::Type> operandTypes, bool isVarArg = false) const;

  /// Generate code that looks like "if then with optional else" at LLVM.
  /// The following prototype code will be generated:
  /// ```
  /// llvm.condBr cond, ^thenBlock, ^elseBlock
  /// ^thenBlock:
  ///   thenBody
  /// ^elseBlock:
  ///   elseBody
  /// ^mainBlock
  ///   ...
  /// ```
  void ifThenElse(valueFuncRef cond, voidFuncRef thenFn,
      voidFuncRef elseFn = nullptr) const;
};

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
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc) {}
  MultiDialectBuilder(const DialectBuilder &db) {}
};

// Recursive class specialized for MathBuilder refereed to as math.
template <class... Ts>
struct MultiDialectBuilder<MathBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), math(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), math(db) {}
  MathBuilder math;
};

// Recursive class specialized for MemRefBuilder refereed to as mem.
template <class... Ts>
struct MultiDialectBuilder<MemRefBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), mem(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), mem(db) {}
  MemRefBuilder mem;
};

// Recursive class specialized for AffineBuilder refereed to as affine.
template <class... Ts>
struct MultiDialectBuilder<AffineBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), affine(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), affine(db) {}
  AffineBuilder affine;
};

// Recursive class specialized for SCFBuilder refereed to as scf.
template <class... Ts>
struct MultiDialectBuilder<SCFBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), scf(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), scf(db) {}
  SCFBuilder scf;
};

// Recursive class specialized for VectorBuilder refereed to as vec.
template <class... Ts>
struct MultiDialectBuilder<VectorBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), vec(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), vec(db) {}
  VectorBuilder vec;
};

// Recursive class specialized for LLVMBuilder refereed to as llvm.
template <class... Ts>
struct MultiDialectBuilder<LLVMBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), llvm(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), llvm(db) {}
  LLVMBuilder llvm;
};

// Include template implementations.
#include "DialectBuilder.hpp.inc"

} // namespace onnx_mlir
