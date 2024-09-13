/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- IndexExprDetail.hpp - Index expression details ---------===//
//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file handle index expressions using indices and calculations using
// literals, affine expressions, and values.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_INDEX_EXPR_DETAIL_H
#define ONNX_MLIR_INDEX_EXPR_DETAIL_H

#include "src/Dialect/Mlir/IndexExpr.hpp"

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
  // Initialize a question mark with the default value of ShapedType::kDynamic.
  void initAsQuestionmark(bool isFloat);
  // Initialize a question mark with a given value.
  void initAsQuestionmark(int64_t const val, bool isFloat);
  // Initialize a question mark for an unknown dimension in a Tensor/Memref.
  // This initialization is needed for symbolic shape analysis where each
  // question mark is assigned to a unique value hashed from the given
  // tensorOrMemref and dimension index.
  void initAsQuestionmark(mlir::Value tensorOrMemref, int64_t index);
  void initAsLiteral(int64_t const value, IndexExprKind const kind);
  void initAsLiteral(double const value, IndexExprKind const kind);
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
  bool isFloatType() const;

  bool hasScope() const;
  bool isInCurrentScope() const;
  bool hasAffineExpr() const;
  bool hasValue() const;

  // Getters.
  IndexExprScope &getScope() const;
  IndexExprScope *getScopePtr() const;
  mlir::OpBuilder &getRewriter() const { return getScope().getRewriter(); }
  mlir::Location getLoc() const { return getScope().getLoc(); }
  IndexExprKind getKind() const;
  int64_t getLiteral() const;
  double getFloatLiteral() const;
  int64_t getQuestionmark() const;
  mlir::AffineExpr getAffineExpr();
  void getAffineMapAndOperands(
      mlir::AffineMap &map, llvm::SmallVectorImpl<mlir::Value> &operands);
  mlir::Value getValue();

  // Setters.
  void setLiteral(int64_t val);
  void setLiteral(double val);
  void setLiteral(const IndexExprImpl &obj);

  void debugPrint(const std::string &msg);

private:
  // Init for internal use only.
  void init(bool isDefined, bool isIntLit, bool isFloatLit, IndexExprKind type,
      int64_t const newIntOrFloatLit, mlir::AffineExpr const affineExpr,
      mlir::Value const value);

  // Data.
  IndexExprScope *scope;
  // Defined implies having a valid intLit, affineExpr, or value expression.
  bool defined;
  // Literal implies having a valid intLit; may also have an affineExpr or
  // value. Literal is true for integer and float values.
  bool literal;
  // Represent a float value. Valid only for affine (literal) or non-affine
  // kinds. Question mark kinds are not typed at this time.
  bool isFloat;
  // Type of IndexExpr. Literal are by default affine.
  IndexExprKind kind;
  // Integer/float value for "literal" kind and ID for "question mark" kind.
  union {
    // Integer value of the literal for "literal" kinds for which "isFloat" is
    // false. Question mark ID for "question mark" kids. ID is always negative.
    int64_t intLit;
    // Float value, valid for "literal" kinds for which "isFloat" is true.
    double floatLit;
  };
  // Affine expression, may be defined for literal, symbols, dims, or affine
  // expr. Not available for float values.
  mlir::AffineExpr affineExpr;
  // Value expression, may be defined whenever the IndexExpr is defined.
  mlir::Value value;
};

} // namespace onnx_mlir
#endif
