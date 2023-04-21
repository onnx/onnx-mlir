/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- DialectBuilder.hpp - Helper functions for MLIR dialects -----===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for building MLIR operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

// Please do not add dependences on ONNX or KRNL dialects.
#include "src/Dialect/Mlir/IndexExpr.hpp"

namespace onnx_mlir {

struct DialectBuilder {
  // Constructor for analysis (no code generation, get builder disabled).
  DialectBuilder(mlir::Location loc) : builder(nullptr), location(loc) {}
  // Constructors for code generation.
  DialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : builder(&b), location(loc) {}
  DialectBuilder(const DialectBuilder &db)
      : builder(db.builder), location(db.location) {}
  virtual ~DialectBuilder() {}
  DialectBuilder(DialectBuilder &&) = delete;
  DialectBuilder &operator=(const DialectBuilder &) = delete;
  DialectBuilder &&operator=(const DialectBuilder &&) = delete;

  // Public getters of builder and location.
  mlir::OpBuilder &getBuilder() const { return b(); }
  mlir::OpBuilder *getBuilderPtr() const { return builder; } // Possibly null.
  mlir::Location getLoc() const { return loc(); }

protected:
  // Private getters of builder and location (concise version).
  mlir::OpBuilder &b() const {
    assert(builder && "builder is null");
    return *builder;
  }
  mlir::Location loc() const { return location; }

private:
  mlir::OpBuilder *builder;
  mlir::Location location;
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
  MathBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  MathBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  MathBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~MathBuilder() {}

  // Support for vectors: we provide queries that work regardless of if we have
  // (1) a scalar or (2) a vector of a basic element type.

  // The method belows ignore the vectors part of the type to provide answer on
  // the basic element types alone.
  static bool isIntegerWithVector(mlir::Type elementOrVectorType);
  static bool isUnsignedIntegerWithVector(mlir::Type elementOrVectorType);
  static bool isFloatWithVector(mlir::Type elementOrVectorType);
  // Return the basic element type regardless of if we are given (1) a scalar or
  // (2) a vector of a basic element type.
  static mlir::Type elementTypeWithVector(mlir::Type elementOrVectorType);
  // Return a type of the same vector shape as vectorType with a basic element
  // type of elementType. When vectorType is null, then the returned type is
  // simply a scalar of elementType.
  static mlir::Type getTypeWithVector(
      mlir::VectorType vectorType, mlir::Type elementType);

  mlir::Value abs(mlir::Value val) const;
  mlir::Value add(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value andi(mlir::Value lhs, mlir::Value rhs) const;     // Int only.
  mlir::Value ceil(mlir::Value val) const;                      // Float only.
  mlir::Value ceilDiv(mlir::Value lhs, mlir::Value rhs) const;  // Int only.
  mlir::Value copySign(mlir::Value rem, mlir::Value div) const; // Float only.
  mlir::Value div(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value exp(mlir::Value val) const;                       // Float only.
  mlir::Value exp2(mlir::Value val) const;                      // Float only.
  mlir::Value floor(mlir::Value val) const;                     // Float only.
  mlir::Value floorDiv(mlir::Value lhs, mlir::Value rhs) const; // Int only.
  mlir::Value log(mlir::Value val) const;                       // Float only.
  mlir::Value log2(mlir::Value val) const;                      // Float only.
  mlir::Value mul(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value neg(mlir::Value val) const;
  mlir::Value ori(mlir::Value lhs, mlir::Value rhs) const;  // Int only.
  mlir::Value pow(mlir::Value base, mlir::Value exp) const; // Float only.
  mlir::Value rem(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value sqrt(mlir::Value val) const; // Float only.
  mlir::Value sub(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value xori(mlir::Value lhs, mlir::Value rhs) const; // Int only.

  mlir::Value select(mlir::Value cmp, mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value gt(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value ge(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value lt(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value le(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value eq(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value neq(mlir::Value lhs, mlir::Value rhs) const;
  // Signed versions (index/signless/signed int or float)
  mlir::Value sgt(mlir::Value lhs, mlir::Value rhs) const; // No unsigned.
  mlir::Value sge(mlir::Value lhs, mlir::Value rhs) const; // No unsigned.
  mlir::Value slt(mlir::Value lhs, mlir::Value rhs) const; // No unsigned.
  mlir::Value sle(mlir::Value lhs, mlir::Value rhs) const; // No unsigned.
  // Unsigned versions
  mlir::Value ugt(mlir::Value lhs, mlir::Value rhs) const; // Unsigned int only
  mlir::Value uge(mlir::Value lhs, mlir::Value rhs) const; // Unsigned int only
  mlir::Value ult(mlir::Value lhs, mlir::Value rhs) const; // Unsigned int only
  mlir::Value ule(mlir::Value lhs, mlir::Value rhs) const; // Unsigned int only

  mlir::Value min(mlir::Value lhs, mlir::Value rhs) const;
  mlir::Value max(mlir::Value lhs, mlir::Value rhs) const;

  mlir::Value constant(mlir::Type type, double val) const;
  mlir::Value constantIndex(int64_t val) const;

  mlir::Attribute negativeInfAttr(mlir::Type type) const;
  mlir::Attribute positiveInfAttr(mlir::Type type) const;

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
// Shape Builder
//===----------------------------------------------------------------------===//

struct ShapeBuilder final : DialectBuilder {
  ShapeBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  ShapeBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  ShapeBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~ShapeBuilder() {}

  mlir::Value dim(mlir::Value val, int64_t index) const;
  mlir::Value shapeOf(mlir::Value val) const;
  mlir::Value getExtent(mlir::Value val, int64_t index) const;
};

//===----------------------------------------------------------------------===//
// MemRef Builder with added support for aligned memory
//===----------------------------------------------------------------------===//

// Default alignment attribute for all allocation of memory. On most system, it
// numElems is 16 bytes.
static constexpr int64_t gDefaultAllocAlign = 16;

struct MemRefBuilder final : DialectBuilder {
  MemRefBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  MemRefBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  MemRefBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~MemRefBuilder() {}

  // Constants
  static const int64_t defaultAlign;

  // Info: get static and dynamic size of memory. Return true if static only.
  bool getStaticAndDynamicMemSize(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t &staticSize,
      IndexExpr &dynSize) const;
  bool getStaticAndDynamicMemSize(mlir::MemRefType type,
      llvm::SmallVectorImpl<IndexExpr> &dims, int64_t &staticSize,
      IndexExpr &dynSize) const;

  // Alloc for static shapes without alignment.
  mlir::memref::AllocOp alloc(mlir::MemRefType type) const;
  // Alloc for static/dynamic shapes without alignment.
  mlir::memref::AllocOp alloc(
      mlir::MemRefType type, mlir::ValueRange dynSymbols) const;
  mlir::memref::AllocOp alloc(
      mlir::Value operandOfSameType, mlir::MemRefType type) const;
  mlir::memref::AllocOp alloc(
      mlir::MemRefType type, llvm::SmallVectorImpl<IndexExpr> &dims) const;

  // Alloc for static shapes with alignment.
  // Minimum alignment is gDefaultAllocAlign.
  mlir::memref::AllocOp alignedAlloc(
      mlir::MemRefType type, int64_t align = defaultAlign) const;
  // Alloc for static/dynamic shapes with alignment.
  mlir::memref::AllocOp alignedAlloc(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t align = defaultAlign) const;
  mlir::memref::AllocOp alignedAlloc(mlir::Value operandOfSameType,
      mlir::MemRefType type, int64_t align = defaultAlign) const;
  mlir::memref::AllocOp alignedAlloc(mlir::MemRefType type,
      llvm::SmallVectorImpl<IndexExpr> &dims,
      int64_t align = defaultAlign) const;

  // Alloc for shapes with alignment and padding for safe full SIMD operations.
  // Padding may be added so that every values in the shape may safely be
  // computed by a SIMD operation (or possibly multiple ones when simdUnroll>1).
  // Minimum alignment is gDefaultAllocAlign.
  // Operation does not support layouts at this time.
  //
  // Alloc for static shapes with alignment and SIMD padding.
  mlir::Value alignedAllocWithSimdPadding(mlir::MemRefType type,
      int64_t simdUnroll = 1, int64_t align = defaultAlign) const;
  // Alloc for static/dynamic shapes with alignment and SIMD padding.
  mlir::Value alignedAllocWithSimdPadding(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t simdUnroll = 1,
      int64_t align = defaultAlign) const;
  mlir::Value alignedAllocWithSimdPadding(mlir::Value operandOfSameType,
      mlir::MemRefType type, int64_t simdUnroll = 1,
      int64_t align = defaultAlign) const;
  mlir::Value alignedAllocWithSimdPadding(mlir::MemRefType type,
      llvm::SmallVectorImpl<IndexExpr> &dims, int64_t simdUnroll = 1,
      int64_t align = defaultAlign) const;

  // The alloca instruction allocates memory on the stack frame of the currently
  // executing function, to be automatically released when this function returns
  // to its caller. It is strongly suggested to place alloca instructions
  // outside of a loop.
  mlir::memref::AllocaOp alloca(mlir::MemRefType type) const;
  mlir::memref::AllocaOp alignedAlloca(
      mlir::MemRefType type, int64_t align = defaultAlign) const;

  mlir::memref::DeallocOp dealloc(mlir::Value val) const;

  // Reshapes.
  mlir::memref::ReshapeOp reshape(mlir::MemRefType destType,
      mlir::Value valToReshape, mlir::Value destShapeStoredInMem) const;
  mlir::memref::ReshapeOp reshapeToFlat(mlir::Value valToReshape,
      llvm::SmallVectorImpl<IndexExpr> &nDims, mlir::Value &size1D) const;
  mlir::memref::ReshapeOp reshapeFromFlat(mlir::Value valToReshape,
      llvm::SmallVectorImpl<IndexExpr> &nDims,
      mlir::MemRefType outputType) const;

  // Casts.
  mlir::memref::CastOp cast(
      mlir::Value input, mlir::MemRefType outputType) const;
  mlir::Value reinterpretCast(
      mlir::Value input, llvm::SmallVectorImpl<IndexExpr> &outputDims) const;

  // Does not support layouts at this time. Does only work for values that are
  // then loaded with affine or memref scalar load/store (MLIR limitations).
  mlir::Value collapseShape(mlir::Value input,
      llvm::ArrayRef<mlir::ReassociationIndices> reassociation);

  // Create a view of input value (<byte size>xi8) starting at byteOffset and
  // shaped by outputType.
  mlir::memref::ViewOp view(mlir::Value input, int64_t byteOffset,
      mlir::MemRefType outputType, mlir::ValueRange outputDynSymbols) const;

  // Create a subview of val. Size of 1 => remove that dim.
  mlir::memref::SubViewOp subView(mlir::Value val,
      llvm::SmallVectorImpl<IndexExpr> &offsets, // Offset for each val dims.
      llvm::SmallVectorImpl<IndexExpr> &sizes,   // Sizes for each val dims.
      llvm::SmallVectorImpl<IndexExpr> &strides) // Stride for each val dims.
      const;

  mlir::Value dim(mlir::Value val, int64_t index) const;
  mlir::Value dim(mlir::Value val, mlir::Value index) const;

private:
  mlir::IntegerAttr computeAlignment(int64_t alignment) const;
  void computeDynSymbols(
      mlir::MemRefType type, // Use type to determine dynamic dimensions.
      llvm::SmallVectorImpl<IndexExpr> &dims, // Get dyn syms from index expr.
      llvm::SmallVectorImpl<mlir::Value> &dynSymbols) // Output dim symbols.
      const;
  void computeDynSymbols(
      mlir::Value operandOfSameType, // Extract dyn symbols from this value.
      mlir::MemRefType type, // Use type to determine dynamic dimensions.
      llvm::SmallVectorImpl<mlir::Value> &dynSymbols) // Output dim symbols.
      const;
};

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF) Builder
//===----------------------------------------------------------------------===//

struct SCFBuilder final : DialectBuilder {
  SCFBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  SCFBuilder(mlir::OpBuilder &b, mlir::Location loc) : DialectBuilder(b, loc) {}
  SCFBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~SCFBuilder() {}

  /// Create an if then with optional else. Construct does not generate a result
  /// (unlike some scf::if) and introduces the yields automatically.
  void ifThenElse(mlir::Value cond,
      mlir::function_ref<void(SCFBuilder &createSCF)> thenFn,
      mlir::function_ref<void(SCFBuilder &createSCF)> elseFn = nullptr) const;

  void parallelLoop(mlir::ValueRange lowerBounds, mlir::ValueRange upperBounds,
      mlir::ValueRange steps,
      mlir::function_ref<void(SCFBuilder &, mlir::ValueRange)> bodyFn) const;
  void yield() const;
};

//===----------------------------------------------------------------------===//
// Vector Builder
//===----------------------------------------------------------------------===//

struct VectorBuilder final : DialectBuilder {
  VectorBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  VectorBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  VectorBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~VectorBuilder() {}

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

  // Splat: a single value is copied.
  mlir::Value splat(mlir::VectorType vecType, mlir::Value val) const;
  // Broadcast: possibly a N dim vector is copied to M>N dim vector.
  mlir::Value broadcast(mlir::VectorType vecType, mlir::Value val) const;
  // Shuffle: use mask to determine which value to write to the output.
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
  GenericAffineBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  GenericAffineBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  GenericAffineBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~GenericAffineBuilder() {}

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

  // AffineApplyOp
  mlir::Value apply(mlir::AffineMap map, mlir::ValueRange operands) const;

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

  LLVMBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  LLVMBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  LLVMBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~LLVMBuilder() {}

  // AddOp
  mlir::Value add(mlir::Value lhs, mlir::Value rhs) const;

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

  // Inttoptr
  mlir::Value inttoptr(mlir::Type type, mlir::Value val) const;

  // LoadOp
  mlir::Value load(mlir::Value addr) const;

  // MulOp
  mlir::Value mul(mlir::Value lhs, mlir::Value rhs) const;

  // NullOp
  mlir::Value null(mlir::Type type) const;
  mlir::Value nullI8Ptr() const;

  // Ptrtoint
  mlir::Value ptrtoint(mlir::Type type, mlir::Value val) const;

  // ReturnOp
  void _return(mlir::Value val) const;

  // SExtOp
  mlir::Value sext(mlir::Type type, mlir::Value val) const;

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
  *  VectorBuilder, access field with vec
*/

// Anchor class.
template <class... Ts>
struct MultiDialectBuilder {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : builder(&b), location(loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : builder(db.getBuilderPtr()), location(db.getLoc()) {}

  // Public getters of builder and location.
  mlir::OpBuilder &getBuilder() const {
    assert(builder);
    return *builder;
  }
  mlir::OpBuilder *getBuilderPtr() const { return builder; }
  mlir::Location getLoc() const { return location; }

private:
  mlir::OpBuilder *builder;
  mlir::Location location;
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

// Recursive class specialized for ShapeBuilder refereed to as shape.
template <class... Ts>
struct MultiDialectBuilder<ShapeBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), shape(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), shape(db) {}
  ShapeBuilder shape;
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
