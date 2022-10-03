/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- IndexExprDetail.hpp - Index expression details ---------===//
//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calculations using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/Mlir/IndexExpr.hpp"

extern int64_t IndexExpr_gQuestionMarkCounter;

namespace onnx_mlir {

// Implementation of the IndexExpr. In nearly all cases, the value described by
// this data structure is constant. Sole exception is during the reduction
// operations. IndexExpr are simply a pointer to this data structure. This data
// structure is allocated in dynamic memory and resides in the scope. It will
// be automaticaly destroyed at the same time as the scope.

class IndexExprImpl {
public:
  // Public constructor.
  IndexExprImpl();

  // Basic initialization calls.
  void initAsUndefined();
  void initAsQuestionmark();
  void initAsLiteral(int64_t const value, IndexExprKind const kind);
  void initAsKind(mlir::Value const value, IndexExprKind const kind);
  void initAsAffineExpr(mlir::AffineExpr const value);
  // Transformational initialization calls.
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
  bool isInCurrentScope() const;
  bool hasAffineExpr() const;
  bool hasValue() const;

  // Getters
  IndexExprScope &getScope() const;
  IndexExprScope *getScopePtr() const;
  mlir::OpBuilder &getRewriter() const { return getScope().getRewriter(); }
  mlir::Location getLoc() const { return getScope().getLoc(); }
  IndexExprKind getKind() const;
  int64_t getLiteral() const;
  int64_t getQuestionmark() const;
  mlir::AffineExpr getAffineExpr();
  void getAffineMapAndOperands(
      mlir::AffineMap &map, llvm::SmallVectorImpl<mlir::Value> &operands);
  mlir::Value getValue();

  // Data.
  IndexExprScope *scope;
  // Defined implies having a valid intLit, affineExpr, or value expression.
  bool defined;
  // Literal implies having a valid intLit; may also have an affineExpr or
  // value.
  bool literal;
  // Type of IndexExpr. Literal are by default affine.
  IndexExprKind kind;
  // Integer value, valid when "literal" or "question mark" is true. Negative
  // value in case of question mark.
  int64_t intLit;
  // Affine expression, may be defined for literal, symbols, dims, or affine
  // expr.
  mlir::AffineExpr affineExpr;
  // Value expression, may be defined whenever the IndexExpr is defined.
  mlir::Value value;

private:
  // Init for internal use only.
  void init(bool isDefined, bool isIntLit, IndexExprKind type,
      int64_t const intLit, mlir::AffineExpr const affineExpr,
      mlir::Value const value);
};

} // namespace onnx_mlir
