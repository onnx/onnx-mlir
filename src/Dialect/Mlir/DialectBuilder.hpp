/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- DialectBuilder.hpp - Helper functions for MLIR dialects -----===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions for building MLIR operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DIALECT_BUILDER_MLIR_H
#define ONNX_MLIR_DIALECT_BUILDER_MLIR_H

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
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"

#include <functional>

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
    assert(builder);
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
  static bool isVector(mlir::Value val);
  static bool isVector(mlir::Type type);
  // The method belows ignore the vectors part of the type to provide answer on
  // the basic element types alone.
  static bool isScalarOrVectorInteger(mlir::Value val);
  static bool isScalarOrVectorInteger(mlir::Type elementOrVectorType);
  static bool isScalarOrVectorUnsignedInteger(mlir::Value val);
  static bool isScalarOrVectorUnsignedInteger(mlir::Type elementOrVectorType);
  static bool isScalarOrVectorFloat(mlir::Value val);
  static bool isScalarOrVectorFloat(mlir::Type elementOrVectorType);
  // Return the basic element type regardless of if we are given (1) a scalar or
  // (2) a vector of a basic element type.
  static mlir::Type elementTypeOfScalarOrVector(mlir::Value val);
  static mlir::Type elementTypeOfScalarOrVector(mlir::Type elementOrVectorType);
  // Return a type of the same vector shape as vectorType with a basic element
  // type of elementType. When vectorType is not a vector, then the returned
  // type is simply a scalar of elementType. ElementType should not be a scalar
  // type.
  static mlir::Type getTypeWithVector(
      mlir::Type vectorType, mlir::Type elementType);

  // "B" below indicates that the operation will splat scalar values if one of
  // the input value is itself a vector.

  // "B" below indicates that the operation will splat scalar values if one of
  // the input value is itself a vector.

  // "B" below indicates that the operation will splat scalar values if one of
  // the input value is itself a vector.

  mlir::Value abs(mlir::Value val) const;
  mlir::Value add(mlir::Value lhs, mlir::Value rhs) const;     // B.
  mlir::Value andi(mlir::Value lhs, mlir::Value rhs) const;    // B/Int only.
  mlir::Value ceil(mlir::Value val) const;                     // Float only.
  mlir::Value ceilDiv(mlir::Value lhs, mlir::Value rhs) const; // B/Int only.
  mlir::Value clip(mlir::Value val, mlir::Value lb, mlir::Value ub) const; // B.
  mlir::Value copySign(mlir::Value rem, mlir::Value div) const; // B/Float only.
  mlir::Value div(mlir::Value lhs, mlir::Value rhs) const;      // B.
  mlir::Value erf(mlir::Value val) const;
  mlir::Value exp(mlir::Value val) const;                       // Float only.
  mlir::Value exp2(mlir::Value val) const;                      // Float only.
  mlir::Value floor(mlir::Value val) const;                     // Float only.
  mlir::Value floorDiv(mlir::Value lhs, mlir::Value rhs) const; // B/Int only.
  mlir::Value fma(
      mlir::Value lhs, mlir::Value rhs, mlir::Value acc) const; // B.
  mlir::Value log(mlir::Value val) const;                       // Float only.
  mlir::Value log2(mlir::Value val) const;                      // Float only.
  mlir::Value mul(mlir::Value lhs, mlir::Value rhs) const;      // B.
  mlir::Value neg(mlir::Value val) const;
  mlir::Value ori(mlir::Value lhs, mlir::Value rhs) const;  // B/Int only.
  mlir::Value pow(mlir::Value base, mlir::Value exp) const; // B/Float only.
  mlir::Value rem(mlir::Value lhs, mlir::Value rhs) const;  // B.
  mlir::Value round(mlir::Value) const;                     // Float only.
  mlir::Value roundEven(mlir::Value) const;                 // Float only.
  mlir::Value roundEvenEmulation(mlir::Value) const;        // Float only.
  mlir::Value sqrt(mlir::Value val) const;                  // Float only.
  mlir::Value sub(mlir::Value lhs, mlir::Value rhs) const;  // B.
  mlir::Value tanh(mlir::Value val) const;                  // Float only.
  mlir::Value xori(mlir::Value lhs, mlir::Value rhs) const; // B/Int only.

  mlir::Value select(
      mlir::Value cmp, mlir::Value trueVal, mlir::Value valseVal) const; // B.
  mlir::Value gt(mlir::Value lhs, mlir::Value rhs) const;                // B.
  mlir::Value ge(mlir::Value lhs, mlir::Value rhs) const;                // B.
  mlir::Value lt(mlir::Value lhs, mlir::Value rhs) const;                // B.
  mlir::Value le(mlir::Value lhs, mlir::Value rhs) const;                // B.
  mlir::Value eq(mlir::Value lhs, mlir::Value rhs) const;                // B.
  mlir::Value neq(mlir::Value lhs, mlir::Value rhs) const;               // B.
  // Signed versions (index/signless/signed int or float)
  mlir::Value sgt(mlir::Value lhs, mlir::Value rhs) const; // B/No unsigned.
  mlir::Value sge(mlir::Value lhs, mlir::Value rhs) const; // B/No unsigned.
  mlir::Value slt(mlir::Value lhs, mlir::Value rhs) const; // B/No unsigned.
  mlir::Value sle(mlir::Value lhs, mlir::Value rhs) const; // B/No unsigned.
  // Unsigned versions
  mlir::Value ugt(mlir::Value lhs, mlir::Value rhs) const; // B/Unsigned only.
  mlir::Value uge(mlir::Value lhs, mlir::Value rhs) const; // B/Unsigned only.
  mlir::Value ult(mlir::Value lhs, mlir::Value rhs) const; // B/Unsigned only.
  mlir::Value ule(mlir::Value lhs, mlir::Value rhs) const; // B/Unsigned only.

  mlir::Value min(mlir::Value lhs, mlir::Value rhs) const; // B.
  mlir::Value max(mlir::Value lhs, mlir::Value rhs) const; // B.

  mlir::Value constant(mlir::Type type, double val) const;
  mlir::Value constantIndex(int64_t val) const;

  mlir::TypedAttr negativeInfAttr(mlir::Type type) const;
  mlir::TypedAttr positiveInfAttr(mlir::Type type) const;

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
  mlir::Value cast(mlir::Type destType, mlir::Value val) const; // B.
  mlir::Value castToIndex(mlir::Value val) const;

  // Add indexOffsets to the least significant indices. So if indices are (i, j,
  // k, l) and offsets are (K, L), the results will be (i, j, k+K, l+L).
  void addOffsetToLeastSignificant(mlir::ValueRange indices,
      mlir::ValueRange offsets,
      llvm::SmallVectorImpl<mlir::Value> &computedIndices) const;
  void addOffsetToLeastSignificant(mlir::ArrayRef<IndexExpr> indices,
      mlir::ValueRange offsets,
      llvm::SmallVectorImpl<mlir::Value> &computedIndices) const;
  // Perform splat to match (see below), accepting up to 3 values at most.
  void splatToMatch(llvm::SmallVectorImpl<mlir::Value> &vals) const;

private:
  mlir::Value createArithCmp(
      mlir::Value lhs, mlir::Value rhs, mlir::arith::CmpIPredicate pred) const;
  mlir::Value createArithCmp(
      mlir::Value lhs, mlir::Value rhs, mlir::arith::CmpFPredicate pred) const;
  mlir::Value castToSignless(mlir::Value source, int64_t width) const;
  mlir::Value castToUnsigned(mlir::Value source, int64_t width) const;

  // If any of the first, second, or third values are vector types, splat the
  // other ones to the same VL. Return true if one or more values were splatted.
  bool splatToMatch(mlir::Value &first, mlir::Value &second) const;
  bool splatToMatch(
      mlir::Value &first, mlir::Value &second, mlir::Value &third) const;
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

  mlir::Value fromExtents(mlir::ValueRange extents) const;
  mlir::Value getExtent(mlir::Value val, int64_t index) const;
  mlir::Value toExtentTensor(mlir::Type type, mlir::Value shape) const;
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

  // Common load/store interface (krnl/affine/memref)
  // Add offsets (if any) to the least significant memref dims.
  mlir::Value load(mlir::Value memref, mlir::ValueRange indices = {},
      mlir::ValueRange offsets = {}) const;
  mlir::Value loadIE(mlir::Value memref, mlir::ArrayRef<IndexExpr> indices = {},
      mlir::ValueRange offsets = {}) const;
  void store(mlir::Value val, mlir::Value memref, mlir::ValueRange indices = {},
      mlir::ValueRange offsets = {}) const;
  void storeIE(mlir::Value val, mlir::Value memref,
      mlir::ArrayRef<IndexExpr> indices = {},
      mlir::ValueRange offsets = {}) const;

  // Info: get static and dynamic size of memory in number of elementary type.
  // Array of vector types are not supported at this time.
  //
  // If range r>0: include dims in range [ 0, min(r, rank) ).
  // Range r==0: noop, look at no data.
  // If r<=: include dims from [ max(0, r+rank) to rank ).
  //
  // Default value includes the entire range. Return true if static only within
  // the specified range.
  bool getStaticAndDynamicMemSize(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t &staticSize, IndexExpr &dynSize,
      int64_t range = 1000) const;
  bool getStaticAndDynamicMemSize(mlir::MemRefType type, DimsExprRef dims,
      int64_t &staticSize, IndexExpr &dynSize, int64_t range = 1000) const;
  // Same as above, but does not track of dynamic size.
  static bool getStaticMemSize(
      mlir::MemRefType type, int64_t &staticSize, int64_t range = 1000);

  // Alloc for static shapes without alignment.
  mlir::memref::AllocOp alloc(mlir::MemRefType type) const;
  // Alloc for static/dynamic shapes without alignment.
  mlir::memref::AllocOp alloc(
      mlir::MemRefType type, mlir::ValueRange dynSymbols) const;
  mlir::memref::AllocOp alloc(
      mlir::Value operandOfSameType, mlir::MemRefType type) const;
  mlir::memref::AllocOp alloc(mlir::MemRefType type, DimsExprRef dims) const;

  // Alloc for static shapes with alignment.
  // Minimum alignment is gDefaultAllocAlign.
  mlir::memref::AllocOp alignedAlloc(
      mlir::MemRefType type, int64_t align = defaultAlign) const;
  // Alloc for static/dynamic shapes with alignment.
  mlir::memref::AllocOp alignedAlloc(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t align = defaultAlign) const;
  mlir::memref::AllocOp alignedAlloc(mlir::Value operandOfSameType,
      mlir::MemRefType type, int64_t align = defaultAlign) const;
  mlir::memref::AllocOp alignedAlloc(mlir::MemRefType type, DimsExprRef dims,
      int64_t align = defaultAlign) const;

  // Alloc for shapes with alignment and padding for safe full SIMD
  // operations. Padding may be added so that every values in the shape may
  // safely be computed by a SIMD operation (or possibly multiple ones if
  // unrollSIMD>1). Minimum alignment is gDefaultAllocAlign. Operation does
  // not support layouts at this time.
  //
  // Alloc for static shapes with alignment and SIMD padding.
  mlir::Value alignedAllocWithSimdPadding(mlir::MemRefType type, int64_t VL = 1,
      int64_t align = defaultAlign) const;
  // Alloc for static/dynamic shapes with alignment and SIMD padding.
  mlir::Value alignedAllocWithSimdPadding(mlir::MemRefType type,
      mlir::ValueRange dynSymbols, int64_t VL = 1,
      int64_t align = defaultAlign) const;
  mlir::Value alignedAllocWithSimdPadding(mlir::Value operandOfSameType,
      mlir::MemRefType type, int64_t VL = 1,
      int64_t align = defaultAlign) const;
  mlir::Value alignedAllocWithSimdPadding(mlir::MemRefType type,
      DimsExprRef dims, int64_t VL = 1, int64_t align = defaultAlign) const;

  // The alloca instruction allocates memory on the stack frame of the
  // currently executing function, to be automatically released when this
  // function returns to its caller. It is strongly suggested to place alloca
  // instructions outside of a loop.
  //
  // When possible, DO NOT USE ALLOCA except for a few scalars.
  //
  mlir::memref::AllocaOp alloca(mlir::MemRefType type) const;
  mlir::memref::AllocaOp alignedAlloca(
      mlir::MemRefType type, int64_t align = defaultAlign) const;

  mlir::memref::DeallocOp dealloc(mlir::Value val) const;

  // Reshapes.
  mlir::memref::ReshapeOp reshape(mlir::MemRefType outputType,
      mlir::Value valToReshape, mlir::Value outputShapeStoredInMem) const;
  // Reshape to dimensions passed by destDims. Will create data-structure to
  // hold the dims, save into it, and the perform the actual reshape.
  mlir::memref::ReshapeOp reshape(
      DimsExpr &outputDims, mlir::Value valToReshape) const;
  // Flatten innermost dimensions of a MemRef. User provide the value to
  // reshape (valToReshape), its dims (dims), and the number of innermost
  // loops to collapse (dimsToFlatten). The function computes the new
  // flattened dimensions (flattenDims) and return the flattened value. Values
  // of dimsToFlatten are in the [1, rank of input] range. Legal only on types
  // with identity layouts.
  mlir::Value reshapeToFlatInnermost(mlir::Value valToReshape, DimsExprRef dims,
      DimsExpr &flattenDims, int64_t dimsToFlatten) const;
  // Flatten to a 2D MemRef, with outer dim including outermost dim to axis
  // -1, and inner dim including the remaining innermost dims. Values of axis
  // are in the [1, rank of input) range. Negative axis values are taken from
  // the back. Legal only on types with identity layouts.
  mlir::Value reshapeToFlat2D(mlir::Value valToReshape, DimsExprRef dims,
      DimsExpr &flattenDims, int64_t axis) const;
  // Perform the reverse operation; given a flattened value, unflatten it by
  // giving the function its original unflattened dimensions (outputDims) and
  // type (outputType). Legal only on types with identity layouts.
  mlir::memref::ReshapeOp reshapeFromFlat(mlir::Value valToReshape,
      DimsExpr &outputDims, mlir::MemRefType outputType) const;

  // Casts.
  mlir::memref::CastOp cast(
      mlir::Value input, mlir::MemRefType outputType) const;
  mlir::Value reinterpretCast(mlir::Value input, DimsExpr &outputDims) const;
  mlir::Value reinterpretCast(
      mlir::Value input, mlir::Value offset, DimsExpr &outputDims) const;

  // Does not support layouts at this time. Does only work for values that are
  // then loaded with affine or memref scalar load/store (MLIR limitations).
  mlir::Value collapseShape(mlir::Value input,
      mlir::ArrayRef<mlir::ReassociationIndices> reassociation);

  // Create a view of input value (<byte size>xi8) starting at byteOffset and
  // shaped by outputType.
  mlir::memref::ViewOp view(mlir::Value input, int64_t byteOffset,
      mlir::MemRefType outputType, mlir::ValueRange outputDynSymbols) const;

  // Create a subview of val.
  mlir::memref::SubViewOp subView(mlir::Value val,
      mlir::ArrayRef<int64_t> offsets, // Offset for each val dims.
      mlir::ArrayRef<int64_t> sizes,   // Sizes for each val dims.
      mlir::ArrayRef<int64_t> strides) // Stride for each val dims.
      const;

  // Create a subview of val.
  mlir::memref::SubViewOp subView(mlir::MemRefType outputType, mlir::Value val,
      mlir::ArrayRef<int64_t> offsets, // Offset for each val dims.
      mlir::ArrayRef<int64_t> sizes,   // Sizes for each val dims.
      mlir::ArrayRef<int64_t> strides) // Stride for each val dims.
      const;

  // Create a subview of val. Size of 1 => remove that dim.
  mlir::memref::SubViewOp subView(mlir::Value val,
      mlir::ArrayRef<IndexExpr> offsets, // Offset for each val dims.
      mlir::ArrayRef<IndexExpr> sizes,   // Sizes for each val dims.
      mlir::ArrayRef<IndexExpr> strides) // Stride for each val dims.
      const;

  mlir::Value dim(mlir::Value val, int64_t index) const;
  mlir::Value dim(mlir::Value val, mlir::Value index) const;

  void prefetchIE(mlir::Value memref, mlir::ArrayRef<IndexExpr> indices,
      bool isWrite, unsigned locality, bool isData = true);
  void prefetch(mlir::Value memref, mlir::ValueRange indices, bool isWrite,
      unsigned locality, bool isData = true);

  // Queries about memory
  static bool isNoneValue(mlir::Value value);
  // Check if "innerDims" innermost dims are scalar (size 1).
  static bool hasOneElementInInnermostDims(mlir::Value value, int64_t innerDim);

private:
  mlir::IntegerAttr computeAlignment(int64_t alignment) const;
  void computeDynSymbols(
      mlir::MemRefType type, // Use type to determine dynamic dimensions.
      DimsExprRef dims,      // Get dyn syms from index expr.
      llvm::SmallVectorImpl<mlir::Value> &dynSymbols) // Output dim symbols.
      const;
  void computeDynSymbols(
      mlir::Value operandOfSameType, // Extract dyn symbols from this value.
      mlir::MemRefType type, // Use type to determine dynamic dimensions.
      llvm::SmallVectorImpl<mlir::Value> &dynSymbols) // Output dim symbols.
      const;
};

//===----------------------------------------------------------------------===//
// Functions definitions for SIMD methods (simdIterate & simdReduce)
//===----------------------------------------------------------------------===//

namespace impl {

// For simdIterate: given a list of inputs, create one output value.
template <class BUILDER>
using SimdIterateBodyFn = std::function<mlir::Value(
    const BUILDER &b, mlir::ArrayRef<mlir::Value> inputVals, int64_t VL)>;

// For simdReduce: take one input & one temp reduction value, and generate the
// new reduction value.
template <class BUILDER>
using SimdReductionBodyFn = std::function<mlir::Value(
    const BUILDER &b, mlir::Value inputVal, mlir::Value tmpVal, int64_t VL)>;

// For simdReduce: take one temp simd reduction value, create a scalar
// reduction, and possibly apply post processing to it (e.g. div by number of
// elements).
//
// For simdReduce2D: only the post processing. Reduction is done before.
template <class BUILDER>
using SimdPostReductionBodyFn = std::function<mlir::Value(
    const BUILDER &b, mlir::Value tmpVal, int64_t VL)>;

// Function used for (nearly) all loops, where there is typically one value in
// the provided ValueRange per loop nest.
template <class BUILDER>
using LoopBodyFn = mlir::function_ref<void(const BUILDER &, mlir::ValueRange)>;

} // namespace impl

//===----------------------------------------------------------------------===//
// Structured Control Flow (SCF) Builder
//===----------------------------------------------------------------------===//

struct SCFBuilder final : DialectBuilder {
  SCFBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  SCFBuilder(mlir::OpBuilder &b, mlir::Location loc) : DialectBuilder(b, loc) {}
  SCFBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~SCFBuilder() {}

  /// Create an if then with optional else. Construct does not generate a
  /// result (unlike some scf::if) and introduces the yields automatically.
  using SCFThenElseBodyFn = mlir::function_ref<void(const SCFBuilder &)>;
  void ifThenElse(mlir::Value cond, SCFThenElseBodyFn thenFn,
      SCFThenElseBodyFn elseFn = nullptr) const;
  // Common loop interface (krnl/affine/scf).
  using SCFLoopBodyFn = impl::LoopBodyFn<SCFBuilder>;
  void forLoopIE(IndexExpr lb, IndexExpr ub, int64_t step, bool useParallel,
      SCFLoopBodyFn bodyFn) const;
  void forLoopsIE(mlir::ArrayRef<IndexExpr> lbs, mlir::ArrayRef<IndexExpr> ubs,
      mlir::ArrayRef<int64_t> steps, mlir::ArrayRef<bool> useParallel,
      SCFLoopBodyFn builderFn) const;
  // Custom interface
  void forLoop(
      mlir::Value lb, mlir::Value ub, int64_t step, SCFLoopBodyFn bodyFn) const;
  void parallelLoops(mlir::ValueRange lbs, mlir::ValueRange ubs,
      mlir::ValueRange steps, SCFLoopBodyFn bodyFn) const;

  void yield() const;

  // Common simd loop interface (krnl/affine/scf).
  // For detailed description, see KrnlBuilder.hpp file.
  using SCFSimdIterateBodyFn = impl::SimdIterateBodyFn<SCFBuilder>;
  void simdIterateIE(IndexExpr lb, IndexExpr ub, int64_t VL, bool fullySimd,
      bool useParallel, mlir::ArrayRef<mlir::Value> inputs,
      mlir::ArrayRef<DimsExpr> inputAFs, mlir::ArrayRef<mlir::Value> outputs,
      mlir::ArrayRef<DimsExpr> outputAFs,
      mlir::ArrayRef<SCFSimdIterateBodyFn> simdIterateBodyList) const;

  // For detailed description, see KrnlBuilder.hpp file.
  using SCFSimdReductionBodyFn = impl::SimdReductionBodyFn<SCFBuilder>;
  using SCFSimdPostReductionBodyFn = impl::SimdPostReductionBodyFn<SCFBuilder>;
  void simdReduceIE(IndexExpr lb, IndexExpr ub, int64_t VL, bool fullySimd,
      mlir::ArrayRef<mlir::Value> inputs, mlir::ArrayRef<DimsExpr> inputAFs,
      mlir::ArrayRef<mlir::Value> temps, mlir::ArrayRef<DimsExpr> tempAFs,
      mlir::ArrayRef<mlir::Value> outputs, mlir::ArrayRef<DimsExpr> outputAFs,
      mlir::ArrayRef<mlir::Value> initVals,
      /* reduction function (simd or scalar) */
      mlir::ArrayRef<SCFSimdReductionBodyFn> simdReductionBodyFnList,
      /* post reduction function (simd to scalar + post processing)*/
      mlir::ArrayRef<SCFSimdPostReductionBodyFn> simdPostReductionBodyFnList)
      const;
  void simdReduce2DIE(IndexExpr lb, IndexExpr ub, int64_t VL, bool fullySimd,
      mlir::Value input, DimsExpr inputAF, mlir::Value tmp, DimsExpr tmpAF,
      mlir::Value output, DimsExpr outputAF, mlir::Value initVal,
      /* reduction functions (simd or scalar) */
      SCFSimdReductionBodyFn reductionBodyFn,
      /* post reduction functions (post processing ONLY)*/
      SCFSimdPostReductionBodyFn postReductionBodyFn) const;
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

  using F2 = std::function<mlir::Value(mlir::Value const, mlir::Value const)>;
  enum CombiningKind { ADD, MUL, MAX, MIN, AND, OR, XOR };

  // Check if two types have compatible shapes (assuming that scalar will be
  // splatted to the proper vector shape),
  static bool compatibleShapes(const mlir::Type t1, const mlir::Type t2);
  // Check that the two types have identical elementary types and shapes.
  static bool compatibleTypes(const mlir::Type t1, const mlir::Type t2);

  // Get the machine SIMD vector length for the given elementary type.
  // This can help guide certain optimizations.
  int64_t getArchVectorLength(const mlir::Type &elementType) const;
  int64_t getArchVectorLength(const mlir::VectorType &vecType) const;
  int64_t getArchVectorLength(mlir::Value vecValue) const;

  // Vector load: memref is expected to be scalar, will load a vector's worth
  // of values: e.g. %result = vector.load %base[%i, %j] :
  // memref<100x100xf32>, vector<8xf32>.
  // Add offsets (if any) to the least significant memref dims.
  mlir::Value load(mlir::VectorType vecType, mlir::Value memref,
      mlir::ValueRange indices = {}, mlir::ValueRange offsets = {}) const;
  mlir::Value loadIE(mlir::VectorType vecType, mlir::Value memref,
      mlir::ArrayRef<IndexExpr> indices = {},
      mlir::ValueRange offsets = {}) const;
  // Vector store: memref can be a scalar, will store the vector values.
  // Add offsets (if any) to the least significant memref dims.
  void store(mlir::Value val, mlir::Value memref, mlir::ValueRange indices = {},
      mlir::ValueRange offsets = {}) const;
  void storeIE(mlir::Value val, mlir::Value memref,
      mlir::ArrayRef<IndexExpr> indices = {},
      mlir::ValueRange offsets = {}) const;

  // Splat: a single value is copied.
  mlir::Value splat(mlir::VectorType vecType, mlir::Value val) const;
  // Broadcast: possibly a N dim vector is copied to M>N dim vector.
  mlir::Value broadcast(mlir::VectorType vecType, mlir::Value val) const;
  // Shuffle: use mask to determine which value to write to the output.
  mlir::Value shuffle(mlir::Value lhs, mlir::Value rhs,
      llvm::SmallVectorImpl<int64_t> &mask) const;
  mlir::Value fma(mlir::Value lhs, mlir::Value rhs, mlir::Value acc) const;

  mlir::Value typeCast(mlir::Type resTy, mlir::Value val) const;

  // Composite functions.
  mlir::Value mergeHigh(mlir::Value lhs, mlir::Value rhs, int64_t step) const;
  mlir::Value mergeLow(mlir::Value lhs, mlir::Value rhs, int64_t step) const;
  mlir::Value reduction(CombiningKind kind, mlir::Value value) const;
  void multiReduction(mlir::ArrayRef<mlir::Value> inputVecArray,
      F2 reductionFct, llvm::SmallVectorImpl<mlir::Value> &outputVecArray);

  // Cast vectors to vectors of different shape (e.g. 1D to 2D and back).
  mlir::Value shapeCast(mlir::VectorType newType, mlir::Value vector) const;
  // Extract and insert 1D vector from/to 2D vector.
  mlir::Value extractFrom2D(mlir::Value vector2D, int64_t position) const;
  mlir::Value insertInto2D(
      mlir::Value vector, mlir::Value vector2D, int64_t position) const;

  // Insert and extract one element (scalar).
  mlir::Value extractElement(mlir::Value vector, int64_t position) const;
  mlir::Value insertElement(
      mlir::Value vector, mlir::Value element, int64_t position) const;

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

  // Common load/store interface (krnl/affine/memref)
  // Add offsets (if any) to the least significant memref dims.
  mlir::Value load(mlir::Value memref, mlir::ValueRange indices = {},
      mlir::ValueRange offsets = {}) const;
  mlir::Value loadIE(mlir::Value memref, mlir::ArrayRef<IndexExpr> indices = {},
      mlir::ValueRange offsets = {}) const;
  void store(mlir::Value val, mlir::Value memref, mlir::ValueRange indices = {},
      mlir::ValueRange offsets = {}) const;
  void storeIE(mlir::Value val, mlir::Value memref,
      mlir::ArrayRef<IndexExpr> indices = {},
      mlir::ValueRange offsets = {}) const;

  mlir::Operation *prefetch(mlir::Value memref, mlir::AffineMap map,
      mlir::ValueRange indices, bool isWrite, unsigned localityHint,
      bool isDataCache = true);

  // Common loop interface (krnl/affine/scf).
  using GenericAffineLoopBodyFn = impl::LoopBodyFn<GenericAffineBuilder>;
  void forLoopIE(IndexExpr lb, IndexExpr ub, int64_t step, bool useParallel,
      GenericAffineLoopBodyFn builderFn) const;
  void forLoopsIE(mlir::ArrayRef<IndexExpr> lbs, mlir::ArrayRef<IndexExpr> ubs,
      mlir::ArrayRef<int64_t> steps, mlir::ArrayRef<bool> useParallel,
      GenericAffineLoopBodyFn builderFn) const;

  // Custom interface
  void forLoopIE(IndexExpr lb, IndexExpr ub, int64_t step,
      GenericAffineLoopBodyFn builderFn) const; // Sequential only.

  // Common simd loop interface (krnl/affine/scf).
  using GenericAffineSimdIterateBodyFn =
      impl::SimdIterateBodyFn<GenericAffineBuilder<LOAD_OP, STORE_OP>>;
  void simdIterateIE(IndexExpr lb, IndexExpr ub, int64_t VL, bool fullySimd,
      bool useParallel, mlir::ArrayRef<mlir::Value> inputs,
      mlir::ArrayRef<DimsExpr> inputAFs, mlir::ArrayRef<mlir::Value> outputs,
      mlir::ArrayRef<DimsExpr> outputAFs,
      mlir::ArrayRef<GenericAffineSimdIterateBodyFn> simdIterateBodyList) const;

  using GenericAffineSimdReductionBodyFn =
      impl::SimdReductionBodyFn<GenericAffineBuilder<LOAD_OP, STORE_OP>>;
  using GenericAffineSimdPostReductionBodyFn =
      impl::SimdPostReductionBodyFn<GenericAffineBuilder<LOAD_OP, STORE_OP>>;
  void simdReduceIE(IndexExpr lb, IndexExpr ub, int64_t VL, bool fullySimd,
      mlir::ArrayRef<mlir::Value> inputs, mlir::ArrayRef<DimsExpr> inputAFs,
      mlir::ArrayRef<mlir::Value> temps, mlir::ArrayRef<DimsExpr> tempAFs,
      mlir::ArrayRef<mlir::Value> outputs, mlir::ArrayRef<DimsExpr> outputAFs,
      mlir::ArrayRef<mlir::Value> initVals,
      /* reduction function (simd or scalar) */
      mlir::ArrayRef<GenericAffineSimdReductionBodyFn> simdReductionBodyFnList,
      /* post reduction function (simd to scalar + post processing)*/
      mlir::ArrayRef<GenericAffineSimdPostReductionBodyFn>
          simdPostReductionBodyFnList) const;
  void simdReduce2DIE(IndexExpr lb, IndexExpr ub, int64_t VL, bool fullySimd,
      mlir::Value input, DimsExpr inputAF, mlir::Value tmp, DimsExpr tmpAF,
      mlir::Value output, DimsExpr outputAF, mlir::Value initVal,
      /* reduction functions (simd or scalar) */
      GenericAffineSimdReductionBodyFn reductionBodyFn,
      /* post reduction functions (post processing ONLY)*/
      GenericAffineSimdPostReductionBodyFn postReductionBodyFn) const;

  // This if then else construct has no arguments to the blocks.
  using GenericAffineThenElseBodyFn =
      mlir::function_ref<void(const GenericAffineBuilder<LOAD_OP, STORE_OP> &)>;
  void ifThenElseIE(IndexExprScope &scope, mlir::ArrayRef<IndexExpr> conditions,
      GenericAffineThenElseBodyFn thenFn,
      GenericAffineThenElseBodyFn elseFn) const;

  // AffineApplyOp
  mlir::Value apply(mlir::AffineMap map, mlir::ValueRange operands) const;

  void yield() const;

private:
  // Support for adding blocks.
  void appendToBlock(mlir::Block *block,
      mlir::function_ref<void(mlir::ValueRange)> builderFn) const;
};

// Affine builder uses affine load and store for memory operations. A later
// definition of AffineBuilderKrnlMem will use Krnl load and store for memory
// operations. We recommend to use AffineBuilderKrnlMem when converting the
// Krnl dialect into the affine dialect.
using AffineBuilder = GenericAffineBuilder<mlir::affine::AffineLoadOp,
    mlir::affine::AffineStoreOp>;

//===----------------------------------------------------------------------===//
// LLVM Builder
//===----------------------------------------------------------------------===//

struct LLVMBuilder final : DialectBuilder {
  using voidFuncRef = mlir::function_ref<void(const LLVMBuilder &createLLVM)>;
  using valueFuncRef =
      mlir::function_ref<mlir::Value(const LLVMBuilder &createLLVM)>;

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
  mlir::Value _alloca(mlir::Type resultType, mlir::Type elementType,
      mlir::Value size, int64_t alignment) const;

  // AndOp
  mlir::Value andi(mlir::Value lhs, mlir::Value rhs) const;

  // BitcastOp
  mlir::Value bitcast(mlir::Type type, mlir::Value val) const;

  // BrOp
  void br(
      mlir::ArrayRef<mlir::Value> destOperands, mlir::Block *destBlock) const;

  // CallOp
  mlir::Value call(mlir::ArrayRef<mlir::Type> resultTypes,
      llvm::StringRef funcName, mlir::ArrayRef<mlir::Value> inputs,
      bool isVarArg = false) const;
  mlir::Value call(mlir::ArrayRef<mlir::Type> resultTypes,
      mlir::FlatSymbolRefAttr funcSymbol, mlir::ArrayRef<mlir::Value> inputs,
      bool isVarArg = false) const;

  // CondBrOp
  void condBr(mlir::Value cond, mlir::Block *trueBlock,
      mlir::ArrayRef<mlir::Value> trueOperands, mlir::Block *falseBlock,
      mlir::ArrayRef<mlir::Value> falseOperands) const;

  // ConstantOp
  mlir::Value constant(mlir::Type type, int64_t val) const;
  mlir::Value constant(mlir::Type type, double val) const;

  // ExtractElementOp
  mlir::Value extractElement(
      mlir::Type resultType, mlir::Value container, int64_t position) const;

  // ExtractValueOp
  mlir::Value extractValue(mlir::Type resultType, mlir::Value container,
      mlir::ArrayRef<int64_t> position) const;

  // FuncOp (assume non-variadic functions, otherwise add support like in
  // seen in `call` in this file).
  mlir::LLVM::LLVMFuncOp func(llvm::StringRef name, mlir::Type type,
      bool createUniqueFunc = false) const;

  // GEPOp
  mlir::Value getElemPtr(mlir::Type resultType, mlir::Type elemType,
      mlir::Value base, mlir::ArrayRef<mlir::LLVM::GEPArg> indices) const;

  // GlobalOp
  mlir::LLVM::GlobalOp globalOp(mlir::Type resultType, bool isConstant,
      mlir::LLVM::Linkage, llvm::StringRef name, mlir::Attribute attr,
      uint64_t alignment = 0, bool uniqueName = true) const;

  // ICmpOp
  mlir::Value icmp(
      mlir::LLVM::ICmpPredicate cond, mlir::Value lhs, mlir::Value rhs) const;

  // InsertElementOp
  mlir::Value insertElement(
      mlir::Value vec, mlir::Value val, int64_t position) const;

  // InsertValueOp
  mlir::Value insertValue(mlir::Type resultType, mlir::Value container,
      mlir::Value val, mlir::ArrayRef<int64_t> position) const;

  // Inttoptr
  mlir::Value inttoptr(mlir::Type type, mlir::Value val) const;

  // LShrOp
  mlir::Value lshr(mlir::Value lhs, mlir::Value rhs) const;

  // LoadOp
  mlir::Value load(mlir::Type elementType, mlir::Value addr) const;

  // MulOp
  mlir::Value mul(mlir::Value lhs, mlir::Value rhs) const;

  // NullOp
  mlir::Value null(mlir::Type type) const;

  // OrOp
  mlir::Value ori(mlir::Value lhs, mlir::Value rhs) const;

  // Ptrtoint
  mlir::Value ptrtoint(mlir::Type type, mlir::Value val) const;

  // ReturnOp
  void _return() const;
  void _return(mlir::Value val) const;

  // SelectOp
  mlir::Value select(mlir::Value cmp, mlir::Value lhs, mlir::Value rhs) const;

  // SExtOp
  mlir::Value sext(mlir::Type type, mlir::Value val) const;

  // ShlOp
  mlir::Value shl(mlir::Value lhs, mlir::Value rhs) const;

  // StoreOp
  void store(mlir::Value val, mlir::Value addr) const;

  // TruncOp
  mlir::Value trunc(mlir::Type type, mlir::Value val) const;

  // ZExtOp
  mlir::Value zext(mlir::Type type, mlir::Value val) const;

  //===--------------------------------------------------------------------===//
  // Helper functions
  //===--------------------------------------------------------------------===//

  // Get or insert a function declaration at the beginning of the module.
  mlir::FlatSymbolRefAttr getOrInsertSymbolRef(mlir::ModuleOp module,
      llvm::StringRef symName, mlir::Type resultType,
      mlir::ArrayRef<mlir::Type> operandTypes, bool isVarArg = false) const;

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

  /// Postfix a symbol with the value of `onnx-mlir.symbol-postfix` in the
  /// module attribute.
  static inline std::string SymbolPostfix(
      mlir::ModuleOp &module, std::string symbol) {
    std::string postfix = "";
    if (mlir::StringAttr postfixAttr = module->getAttrOfType<mlir::StringAttr>(
            "onnx-mlir.symbol-postfix")) {
      if (!postfixAttr.getValue().empty())
        postfix = "_" + postfixAttr.getValue().lower();
    }
    return symbol + postfix;
  }

private:
  // Support for calling vararg functions; add the necessary type.
  void handleVarArgCall(mlir::LLVM::CallOp &callOp,
      mlir::ArrayRef<mlir::Type> resultTypes,
      mlir::ArrayRef<mlir::Value> inputs) const;
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
#ifndef ONNX_MLIR_DIALECT_BUILDER_MLIR_INC
#include "DialectBuilder.hpp.inc"
#endif

} // namespace onnx_mlir
#endif
