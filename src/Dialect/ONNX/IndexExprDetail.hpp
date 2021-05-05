/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------IndexExprDetail.hpp - Index expression details---------===
////
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calculations using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/IndexExpr.hpp"

namespace mlir {

// Implementation of the IndexExpr. In nearly all cases, the value described by
// this data structure is constant. Sole exception is during the reduction
// operations. IndexExpr are simply a pointer to this data structure. This data
// structure is allocated in dynamic memory and resides in the scope. It will
// be automaticaly destroyed at the same time as the scope.

struct IndexExprImpl {
  // Public constructor.
  IndexExprImpl();

  // Basic initialization calls.
  void initAsUndefined();
  void initAsQuestionmark();
  void initAsLiteral(int64_t const value, IndexExprKind const kind);
  void initAsKind(Value const value, IndexExprKind const kind);
  void initAsAffineExpr(AffineExpr const value);
  // Transformative initialization calls.
  void initAsKind(IndexExprImpl const *expr, IndexExprKind const kind);

  // Copy.
  void copy(IndexExprImpl const *other);

  // Queries
  bool isShapeInferencePass() const;
  bool isDefined() const;
  bool isLiteral() const;
  bool isQuestionmark() const;
  bool isAffine() const;
  bool isSymbol() const;
  bool isDim() const;
  bool isPredType() const;
  bool isIndexType() const;

  bool hasScope() const;
  bool hasAffineExpr() const;
  bool hasValue() const;

  // Getters
  IndexExprScope &getScope() const;
  IndexExprScope *getScopePtr() const;
  OpBuilder &getRewriter() const { return getScope().getRewriter(); }
  Location getLoc() const { return getScope().getLoc(); }
  IndexExprKind getKind() const;
  int64_t getLiteral() const;
  AffineExpr getAffineExpr();
  void getAffineMapAndOperands(
      AffineMap &map, SmallVectorImpl<Value> &operands);
  Value getValue();

  // Data.
  IndexExprScope *scope;
  // Defined implies having a valid intLit, affineExpr, or value expression.
  bool defined;
  // Type of IndexExpr. Literal are by default affine.
  IndexExprKind kind;
  // Literal implies having a valid intLit; may also have an affineExpr or
  // value.
  bool literal;
  // Integer value, valid when "literal" is true.
  int64_t intLit;
  // Affine expression, may be defined for literal, symbols, dims, or affine
  // expr.
  AffineExpr affineExpr;
  // Value expression, may be defined whenever the IndexExpr is defined.
  Value value;

private:
  // Init for internal use only.
  void init(bool isDefined, bool isIntLit, IndexExprKind type,
      int64_t const intLit, AffineExpr const affineExpr, Value const value);
};

} // namespace mlir
