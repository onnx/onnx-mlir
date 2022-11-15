/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ IndexExprBuilder.cpp - builder for index expressions ----===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file has support for building Index Expressions from common MLIR objects
// such as MemRef/Tensor shapes, scalar or 1 dimension arrays for integers,
// attributes...
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "llvm/ADT/BitVector.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

using namespace mlir;

namespace {

// Local helper.
static bool hasShapeAndRank(Value val) {
  ShapedType shapedType = val.getType().dyn_cast_or_null<ShapedType>();
  return shapedType && shapedType.hasRank();
}

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// IndexShapeBuilder
//===----------------------------------------------------------------------===//

// Get lit from array attribute.
uint64_t IndexExprBuilder::getSize(mlir::ArrayAttr arrayAttr) {
  return arrayAttr.size();
}

IndexExpr IndexExprBuilder::getLiteral(mlir::ArrayAttr arrayAttr, uint64_t i) {
  uint64_t size = arrayAttr.size();
  if (i >= size)
    return UndefinedIndexExpr();
  int64_t val = (arrayAttr.getValue()[i]).cast<IntegerAttr>().getInt();
  return LiteralIndexExpr(val);
}

IndexExpr IndexExprBuilder::getLiteral(
    mlir::ArrayAttr arrayAttr, uint64_t i, int64_t defaultLiteral) {
  IndexExpr indexExpr = getLiteral(arrayAttr, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LiteralIndexExpr(defaultLiteral) : indexExpr;
}

// Get symbol from operands.
uint64_t IndexExprBuilder::getSize(Value scalarOr1DArrayIntValue) {
  assert(hasShapeAndRank(scalarOr1DArrayIntValue) &&
         "expected shaped type with rank");
  ShapedType shapeType = scalarOr1DArrayIntValue.getType().cast<ShapedType>();
  // Find shaped type size (rank of 0 is scalar).
  uint64_t rank = shapeType.getRank();
  assert(rank < 2 && "expected a scalar or a 1 dimension array of int values");
  return (rank == 0) ? 1 : shapeType.getShape()[0];
}

IndexExpr IndexExprBuilder::getSymbol(
    Value scalarOr1DArrayIntValue, uint64_t i) {
  uint64_t size = getSize(scalarOr1DArrayIntValue);
  if (i >= size)
    return UndefinedIndexExpr();
  // If our scalar array is a constant, return it.
  if (DenseElementsAttr attrArray = getConst(scalarOr1DArrayIntValue)) {
    auto attrVal = attrArray.getValues<Attribute>()[ArrayRef<uint64_t>({i})];
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    return LiteralIndexExpr(attrInt);
  }
  // If our scalar array is not a constant; we have a questionmark.
  if (Value val = getVal(scalarOr1DArrayIntValue, i))
    return SymbolIndexExpr(val);
  else
    return QuestionmarkIndexExpr();
}

IndexExpr IndexExprBuilder::getSymbol(
    Value scalarOr1DArrayIntValue, uint64_t i, int64_t defaultLiteral) {
  IndexExpr indexExpr = getSymbol(scalarOr1DArrayIntValue, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LiteralIndexExpr(defaultLiteral) : indexExpr;
}

bool IndexExprBuilder::getSymbols(
    Value scalarOr1DArrayIntValue, IndexExprList &list, int64_t listSize) {
  list.clear();
  uint64_t size = getSize(scalarOr1DArrayIntValue);
  if (listSize == -1) // Meaning pick up the full size of the list.
    listSize = size;
  else if ((uint64_t)listSize > size) // Requesting more elements than avail.
    return false;
  for (uint64_t i = 0; i < (uint64_t)listSize; ++i) {
    IndexExpr indexExpr = getSymbol(scalarOr1DArrayIntValue, i);
    assert(!indexExpr.isUndefined() && "expected defined index expr");
    list.emplace_back(indexExpr);
  }
  return true;
}

// Get info from tensor/memref shape.
bool IndexExprBuilder::isShapeCompileTimeConstant(
    Value tensorOrMemrefValue, uint64_t i) {
  return getShape(tensorOrMemrefValue, i) != -1;
}

bool IndexExprBuilder::isShapeCompileTimeConstant(Value tensorOrMemrefValue) {
  uint64_t rank = getShapeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    if (!isShapeCompileTimeConstant(tensorOrMemrefValue, i))
      return false;
  return true;
}

uint64_t IndexExprBuilder::getShapeRank(Value tensorOrMemrefValue) {
  assert(
      hasShapeAndRank(tensorOrMemrefValue) && "expected shaped type with rank");
  return tensorOrMemrefValue.getType().dyn_cast_or_null<ShapedType>().getRank();
}

int64_t IndexExprBuilder::getShape(Value tensorOrMemrefValue, uint64_t i) {
  uint64_t rank = getShapeRank(tensorOrMemrefValue);
  assert(i < rank && "expected index smaller than memref rank");
  return tensorOrMemrefValue.getType().cast<ShapedType>().getShape()[i];
}

// Get index expressions from tensor/memref shape.
IndexExpr IndexExprBuilder::getShapeAsLiteral(
    Value tensorOrMemrefValue, uint64_t i) {
  int64_t shape = getShape(tensorOrMemrefValue, i);
  assert(shape != -1 && "expected compile time constant shape");
  return LiteralIndexExpr(shape);
}

IndexExpr IndexExprBuilder::getShapeAsSymbol(
    Value tensorOrMemrefValue, uint64_t i) {
  if (isShapeCompileTimeConstant(tensorOrMemrefValue, i))
    return getShapeAsLiteral(tensorOrMemrefValue, i);
  if (Value val = getShapeVal(tensorOrMemrefValue, i))
    return SymbolIndexExpr(val);
  else
    return QuestionmarkIndexExpr(tensorOrMemrefValue, i);
}

IndexExpr IndexExprBuilder::getShapeAsDim(
    Value tensorOrMemrefValue, uint64_t i) {
  if (isShapeCompileTimeConstant(tensorOrMemrefValue, i))
    return getShapeAsLiteral(tensorOrMemrefValue, i);
  if (Value val = getShapeVal(tensorOrMemrefValue, i))
    return DimIndexExpr(val);
  else
    return QuestionmarkIndexExpr(tensorOrMemrefValue, i);
}

void IndexExprBuilder::getShapeAsLiterals(
    Value tensorOrMemrefValue, IndexExprList &list) {
  list.clear();
  uint64_t rank = getShapeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    list.emplace_back(getShapeAsLiteral(tensorOrMemrefValue, i));
}

void IndexExprBuilder::getShapeAsSymbols(
    Value tensorOrMemrefValue, IndexExprList &list) {
  list.clear();
  uint64_t rank = getShapeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    list.emplace_back(getShapeAsSymbol(tensorOrMemrefValue, i));
}

void IndexExprBuilder::getShapeAsDims(
    Value tensorOrMemrefValue, IndexExprList &list) {
  list.clear();
  uint64_t rank = getShapeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    list.emplace_back(getShapeAsDim(tensorOrMemrefValue, i));
}

} // namespace onnx_mlir
