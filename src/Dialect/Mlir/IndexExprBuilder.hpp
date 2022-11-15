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

struct IndexExprBuilder : DialectBuilder {
  using IndexExprList = llvm::SmallVectorImpl<IndexExpr>;

  IndexExprBuilder() {} // hi alex (empty for analysis)
  IndexExprBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : DialectBuilder(b, loc) {}
  IndexExprBuilder(const DialectBuilder &db) : DialectBuilder(db) {}

  // Get lit from array attribute.
  uint64_t getSize(mlir::ArrayAttr array);
  IndexExpr getLiteral(mlir::ArrayAttr arrayAttr, uint64_t i);
  IndexExpr getLiteral(
      mlir::ArrayAttr arrayAttr, uint64_t i, int64_t defaultLiteral);

  // Get symbol from operands.
  uint64_t getSize(mlir::Value scalarOr1DArrayIntValue);
  IndexExpr getSymbol(mlir::Value scalarOr1DArrayIntValue, uint64_t i);
  IndexExpr getSymbol(
      mlir::Value scalarOr1DArrayIntValue, uint64_t i, int64_t defaultLiteral);
  bool getSymbols(mlir::Value scalarOr1DArrayIntValue, IndexExprList &list,
      int64_t listSize = -1);

  // Get info from tensor/memref shape.
  bool isShapeCompileTimeConstant(mlir::Value tensorOrMemrefValue, uint64_t i);
  bool isShapeCompileTimeConstant(mlir::Value tensorOrMemrefValue);
  uint64_t getShapeRank(mlir::Value tensorOrMemrefValue);
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
