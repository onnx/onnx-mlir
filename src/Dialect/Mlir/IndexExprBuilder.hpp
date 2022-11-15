/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ONNXShapeHelper.hpp - help for shapes ---------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#pragma once

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

//===----------------------------------------------------------------------===//
// IndexShapeBuilder
//===----------------------------------------------------------------------===//

// IndexExprBuilder is used to extract index expressions for computations
// typically related to shapes. This class defines all the algorithms but rely
// on subclass to extract "runtime" values. Methods are provided to return
// literal/symbol/dim index expressions related to operation attributes,
// operation operands, and the shape of operands.

// Recall that literals are compile-time integer values, and symbol and dim are
// runtime values. The difference between symbol/dim related to affine
// expression; symbol is not changing in the given context (e.g. batch size in a
// given loop), and dim are changing (e.g. the loop index inside a given loop).
//
// This class cannot be directly used, and must be refined by subclasses.
//
// A first subclass is IndexExprBuilderForAnalysis and is used during the
// analysis phase; runtime values are described by questionmark index
// expressions.
//
// Other subclasses (e.g. IndexExprBuilderForKrnl) generate Krnl dialect
// operations to generate code that compute runtime values.
//
// Subclasses simply have to define three virtual functions: getConst, getVal,
// and getShape to provide the proper values for the methods defined in this
// class.

struct IndexExprBuilder : DialectBuilder {
  using IndexExprList = llvm::SmallVectorImpl<IndexExpr>;

  // Constructor for analysis (no code generation).
  IndexExprBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  // Constructors for code generation.
  IndexExprBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  IndexExprBuilder(const DialectBuilder &db) : DialectBuilder(db) {}

  //===--------------------------------------------------------------------===//
  // Get literal index expressions from an integer array attributes. Typically
  // used for getting literals out of operation's integer attributes. There is
  // no support for ranks higher than 1 at this time.

  // Get size of array attribute.
  uint64_t getIntArrayAttrSize(mlir::ArrayAttr intArrayAttr);
  // Get literal index expression from the value of an integer array attribute
  // at position i. If out of bound, return an undefined index expression.
  IndexExpr getIntArrayAttrAsLiteral(mlir::ArrayAttr intArrayAttr, uint64_t i);
  // Same as above. If out of bound, return an literal index expression of
  // value defaultVal.
  IndexExpr getIntArrayAttrAsLiteral(
      mlir::ArrayAttr intArrayAttr, uint64_t i, int64_t defaultVal);

  //===--------------------------------------------------------------------===//
  // Get symbol index expressions from a 1D integer array value. When the
  // integer array values are defined by a constant, then literal index
  // expressions are return in place of a symbol index expression. With dynamic
  // values, questionmark index expressions are returned during code analysis
  // phases and symbol index expressions are returned during code generation
  // phases. Note that array of rank 0 are treated as scalars. There is no
  // support for ranks higher than 1 at this time.

  // Get size of array defined by intArrayVal value.
  uint64_t getIntArraySize(mlir::Value intArrayVal);
  // Get a symbol index expression from the integer array defined by intArrayVal
  // at position i. If array is defined by a constant, return a literal index
  // expression. If defined by a runtime value, return questionmark or symbol
  // index expressions depending on the phase. If out of bound, return an
  // undefined index expressions.
  IndexExpr getIntArrayAsSymbol(mlir::Value intArrayVal, uint64_t i);
  // Same as above; if out of bound, return a literal index expression of value
  // defaultVal.
  IndexExpr getIntArrayAsSymbol(
      mlir::Value intArrayVal, uint64_t i, int64_t defaultVal);
  // Same as above, but get a list of up to listSize values. Assert when
  // listSize exceed the array bounds.
  void getIntArrayAsSymbols(
      mlir::Value intArrayVal, IndexExprList &list, int64_t listSize = -1);

  //===--------------------------------------------------------------------===//
  // Get info from tensor/memref shape. Return literal index expressions when
  // shape is known at compile time. Returns questionmark for runtime shapes
  // during analysis phases. Returns symbol or dim index expressions for runtime
  // shapes during code generation phases. Asserts when requesting out of bound
  // shapes. Works on tensors and memrefs.

  // Return true if shape is known at compile time, i.e. is a literal value.
  bool isLiteralShape(mlir::Value tensorOrMemrefValue, uint64_t i);
  // Return true if the entire shape is known at compile time.
  bool isLiteralShape(mlir::Value tensorOrMemrefValue);
  // Get rank of shape (which is also the size of the shape vector).
  uint64_t getShapeRank(mlir::Value tensorOrMemrefValue);
  // Get the raw shape (as integer, -1 when runtime value).
  int64_t getShape(mlir::Value tensorOrMemrefValue, uint64_t i);

  // Get index expressions from tensor/memref shape.
  IndexExpr getShapeAsLiteral(mlir::Value tensorOrMemrefValue, uint64_t i);
  IndexExpr getShapeAsSymbol(mlir::Value tensorOrMemrefValue, uint64_t i);
  IndexExpr getShapeAsDim(mlir::Value tensorOrMemrefValue, uint64_t i);
  void getShapeAsLiterals(mlir::Value tensorOrMemrefValue, IndexExprList &list);
  void getShapeAsSymbols(mlir::Value tensorOrMemrefValue, IndexExprList &list);
  void getShapeAsDims(mlir::Value tensorOrMemrefValue, IndexExprList &list);

protected:
  virtual mlir::DenseElementsAttr getConst(mlir::Value value) = 0;
  virtual mlir::Value getVal(
      mlir::Value scalarOr1DArrayIntValue, uint64_t i) = 0;
  virtual mlir::Value getShapeVal(
      mlir::Value tensorOrMemrefValue, uint64_t i) = 0;
};

} // namespace onnx_mlir
