/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ONNXShapeHelper.hpp - help for shapes ---------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines a class that enables the scanning of MLIR Values,
// Attributes, and Type Shape to generate IndexExpr. They are the main way to
// generate IndexExpr in onnx-mlir.
//
// The IndexExprBuilder class has virtual functions that needs to be
// instantiated by specialized sub-classes, defined to work in a specific
// dialect. There are currently 3 sub-classes, one for ONNX, KRNL, and
// Stablehlo.
//
// Namely: IndexExprBuilderForAnalysis, IndexExprBuilderForKrnl, and
// IndexExprBuilderForStablehlo
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_INDEX_EXPR_BUILDER_H
#define ONNX_MLIR_INDEX_EXPR_BUILDER_H

#include <utility>

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"

namespace onnx_mlir {

// ===----------------------------------------------------------------------===//
//  IndexExprBuilder
// ===----------------------------------------------------------------------===/

/*
  IndexExprBuilder is used to extract index expressions for computations
  typically related to shapes. This class defines all the algorithms but rely
  on subclass to extract "runtime" values. Methods are provided to return
  literal/symbol/dim index expressions related to operation attributes,
  operation operands, and the shape of operands.

  Recall that literals are compile-time integer values, and symbol and dim are
  runtime values. The difference between symbol/dim related to affine
  expression; symbol is not changing in the given context (e.g. batch size in a
  given loop), and dim are changing (e.g. the loop index inside a given loop).

  This class cannot be directly used, as subclasses must redefine 3 pure virtual
  functions (getConst, getVal, and getShape) to provide the proper values for
  the methods defined in this class.

  A first subclass is IndexExprBuilderForAnalysis and is used during the
  analysis phase; runtime values are described by questionmark index
  expressions.

  Other subclasses (e.g. IndexExprBuilderForKrnl/IndexExprBuilderForStablehlo )
  generate dialect operations (e.g. Krnl/Stablehlo ops) to generate code that
  compute runtime values.
*/

/* Dialect use:
   May generate math conversion ops, plus  what is possibly generated in the
   virtual subclass method implementation for getConst, getVal, getShapeVal.
*/

struct IndexExprBuilder : DialectBuilder {
  // Constructor for analysis (no code generation, will assert if it tries).
  IndexExprBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  // Constructors for code generation.
  IndexExprBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  IndexExprBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~IndexExprBuilder() {}

  using IndexExprList = llvm::SmallVectorImpl<IndexExpr>;

  //===--------------------------------------------------------------------===//
  // Get info about rank and sizes

  // check/assert that value has shape and rank.
  bool hasShapeAndRank(mlir::Value value);
  void assertHasShapeAndRank(mlir::Value value);

  // Get rank of the type defined by value. Expect ranked shaped type.
  uint64_t getShapedTypeRank(mlir::Value value);
  // Get size of 1D array attribute. Expect 1D ranked shaped type.
  int64_t getArraySize(mlir::ArrayAttr arrayAttr);
  // Get size of 1D array defined by arrayVal. Expect 1D ranked shaped type.
  // When staticSizeOnly is false, it may return ShapedType::kDynamic. When
  // staticSizeOnly is true, it will assert if the shape of the array is
  // dynamic.
  int64_t getArraySize(mlir::Value arrayVal, bool staticSizeOnly = false);

  //===--------------------------------------------------------------------===//
  // Get literal index expressions from an array of integer attributes.
  // Typically used for getting literals out of operation's integer attributes.
  // There is no support for ranks higher than 1 at this time.

  // Get literal index expression from the value of an array attribute at
  // position i. When out of bound, return an undefined index expression.
  IndexExpr getIntFromArrayAsLiteral(mlir::ArrayAttr intAttrArray, uint64_t i);
  // Same as above; `outOfBoundVal` literal index expression is returned when
  // out of bound.
  IndexExpr getIntFromArrayAsLiteral(
      mlir::ArrayAttr intAttrArray, uint64_t i, int64_t outOfBoundVal);
  // Same as above, but get a list of up to len values. A length of -1 returns
  // the whole list. Assert when `len` exceed the array bounds.
  void getIntFromArrayAsLiterals(
      mlir::ArrayAttr intAttrArray, IndexExprList &list, int64_t len = -1);
  // Same as above, but get a list of len values, filling in with outOfBoundVal
  // if the actual attribute array does not have sufficient number of values.
  // Expects an actual (i.e.  nonnegative) length.
  void getIntFromArrayAsLiterals(mlir::ArrayAttr intAttrArray,
      int64_t outOfBoundVal, IndexExprList &list, int64_t len);

  //===--------------------------------------------------------------------===//
  // Get symbol/dim index expressions from a scalar or 1D array value. When
  // the values are defined by a constant, then literal index expressions are
  // return in place of a symbol index expression. With dynamic values,
  // questionmark index expressions are returned during code analysis phases and
  // symbol index expressions are returned during code generation phases. Note
  // that array of rank 0 are treated as scalars. Introduce conversions to index
  // type when input is in a different type.
  //
  // There is no support for ranks higher than 1 at this time.  Expects a shaped
  // type with a known rank.

  // Get a symbol/dim index expression defined by `value`.
  IndexExpr getIntAsSymbol(mlir::Value value);
  IndexExpr getIntAsDim(mlir::Value value);
  IndexExpr getFloatAsNonAffine(mlir::Value value);
  // Get a symbol/dim index expression from the int or float array defined by
  // `intArray` / `floatArray` at position `i`.
  // When out of bound, return an undefined index expressions. If array size is
  // known, it can be passed as the arraySize argument. Otherwise (-1), the call
  // will determine it from the intArray value.
  IndexExpr getIntFromArrayAsSymbol(
      mlir::Value intArray, uint64_t i, int64_t arraySize = -1);
  IndexExpr getIntFromArrayAsDim(
      mlir::Value intArray, uint64_t i, int64_t arraySize = -1);
  IndexExpr getFloatFromArrayAsNonAffine(
      mlir::Value floatArray, uint64_t i, int64_t arraySize = -1);
  // Same as above; `outOfBoundVal` literal index expression is returned
  // when out of bound.
  IndexExpr getIntFromArrayAsSymbolWithOutOfBound(
      mlir::Value intArray, uint64_t i, int64_t outOfBoundVal);
  IndexExpr getIntFromArrayAsDimWithOutOfBound(
      mlir::Value intArray, uint64_t i, int64_t outOfBoundVal);
  IndexExpr getFloatFromArrayAsNonAffineWithOutOfBound(
      mlir::Value floatArray, uint64_t i, double outOfBoundVal);
  // Same as above, but get a list of up to len values. A length of -1 returns
  // the whole list. Assert when `len` exceed the array bounds.
  void getIntFromArrayAsSymbols(
      mlir::Value intArrayVal, IndexExprList &list, int64_t len = -1);
  void getIntFromArrayAsDims(
      mlir::Value intArrayVal, IndexExprList &list, int64_t len = -1);
  void getFloatFromArrayAsNonAffine(
      mlir::Value floatArrayVal, IndexExprList &list, int64_t len = -1);

  //===--------------------------------------------------------------------===//
  // Get info from tensor/memref shape. Return literal index expressions when a
  // shape is known at compile time. Returns a questionmark for a runtime shape
  // during analysis phases. Returns a symbol or dim index expression for a
  // runtime shape during code generation phases. Works on tensors and memrefs.
  // Asserts when requesting out of bound shapes.  Expects a shaped type with a
  // known rank.

  // Return true if shape is known at compile time, i.e. is a literal value.
  bool isLiteralShape(mlir::Value tensorOrMemrefValue, uint64_t i);
  // Return true if the entire shape is known at compile time.
  bool isLiteralShape(mlir::Value tensorOrMemrefValue);
  // Get the raw shape (as integer, ShapedType::kDynamic when runtime value).
  int64_t getShape(mlir::Value tensorOrMemrefValue, uint64_t i);

  // Get an index expression from tensor/memref shape.
  IndexExpr getShapeAsLiteral(mlir::Value tensorOrMemrefValue, uint64_t i);
  IndexExpr getShapeAsSymbol(mlir::Value tensorOrMemrefValue, uint64_t i);
  IndexExpr getShapeAsDim(mlir::Value tensorOrMemrefValue, uint64_t i);
  // Get an index expression list from tensor/memref shape.
  void getShapeAsLiterals(mlir::Value tensorOrMemrefValue, IndexExprList &list);
  void getShapeAsSymbols(mlir::Value tensorOrMemrefValue, IndexExprList &list);
  void getShapeAsDims(mlir::Value tensorOrMemrefValue, IndexExprList &list);

  //===--------------------------------------------------------------------===//
  // Index expression helpers for tiled data. In the expressions below:
  //   i:     the index at the start of the tile
  //   block: the size of the tile block
  //   UB:    the upper bound of the index space.

  // Determined if we have a full tile (affine expression compared to >=0)
  IndexExpr isTileFull(IndexExpr i, IndexExpr block, IndexExpr UB);
  // Only if tile is not full, the remaining trip count within the tile.
  IndexExpr partialTileSize(IndexExpr i, IndexExpr block, IndexExpr UB);
  // The trip count within the tile, regardless of if full or partial
  IndexExpr tileSize(IndexExpr i, IndexExpr block, IndexExpr UB);

protected:
  //===--------------------------------------------------------------------===//
  // Subclasses must define these pure virtual functions.

  // Locate an elements attribute associated with the defining op given by
  // value. Return nullptr if none exists.
  virtual mlir::ElementsAttr getConst(mlir::Value value) = 0;
  // Locate/generate a value that represents a value given by the op defining
  // arrayVal at position i in the array. Return nullptr if cannot
  // locate/generate the value.
  virtual mlir::Value getVal(mlir::Value arrayVal, uint64_t i) = 0;
  // Locate/generate a value that represents the integer value of the shape
  // given by a tensor or memref at position i. Return nullptr if cannot
  // locate/generate the value.
  virtual mlir::Value getShapeVal(
      mlir::Value tensorOrMemrefValue, uint64_t i) = 0;

private:
  // Returns a SymbolIndexExpr/DimIndexExpr when makeSymbol is true/false.
  // 'array' element type must match 'isFloat'. If arraySize >=0, use that size.
  // Otherwise, get the size from the array value.
  IndexExpr getValFromArray(mlir::Value array, uint64_t i, bool makeSymbol,
      bool isFloat, int64_t arraySize);
};

} // namespace onnx_mlir
#endif
