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

//===----------------------------------------------------------------------===//
// Get literals from integer array attribute.

uint64_t IndexExprBuilder::getIntArrayAttrSize(ArrayAttr intArrayAttr) {
  return intArrayAttr.size();
}

IndexExpr IndexExprBuilder::getIntArrayAttrAsLiteral(
    ArrayAttr intArrayAttr, uint64_t i) {
  uint64_t size = intArrayAttr.size();
  if (i >= size)
    return UndefinedIndexExpr();
  int64_t val = (intArrayAttr.getValue()[i]).cast<IntegerAttr>().getInt();
  return LiteralIndexExpr(val);
}

IndexExpr IndexExprBuilder::getIntArrayAttrAsLiteral(
    ArrayAttr intArrayAttr, uint64_t i, int64_t defaultVal) {
  IndexExpr indexExpr = getIntArrayAttrAsLiteral(intArrayAttr, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LiteralIndexExpr(defaultVal) : indexExpr;
}

//===----------------------------------------------------------------------===//
// Get symbols from value defined by intArrayVal.

uint64_t IndexExprBuilder::getIntArrayRank(Value intArrayVal) {
  assert(hasShapeAndRank(intArrayVal) && "expected shaped type with rank");
  ShapedType shapeType = intArrayVal.getType().cast<ShapedType>();
  // Find shaped type size (rank of 0 is scalar).
  return shapeType.getRank();
}

uint64_t IndexExprBuilder::getIntArraySize(Value intArrayVal) {
  uint64_t rank = getIntArrayRank(intArrayVal);
  assert(rank < 2 && "expected a scalar or a 1 dimension array of int values");
  if (rank == 0)
    return 1;
  ShapedType shapeType = intArrayVal.getType().cast<ShapedType>();
  return shapeType.getShape()[0];
}

IndexExpr IndexExprBuilder::getIntArrayAsSymbol(Value intArrayVal, uint64_t i) {
  uint64_t size = getIntArraySize(intArrayVal);
  if (i >= size)
    return UndefinedIndexExpr();
  // If our scalar array is a constant, return it.
  if (DenseElementsAttr attrArray = getConst(intArrayVal)) {
    auto attrVal = attrArray.getValues<Attribute>()[ArrayRef<uint64_t>({i})];
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    return LiteralIndexExpr(attrInt);
  }
  // If our scalar array is not a constant; we have a questionmark.
  if (Value val = getVal(intArrayVal, i))
    return SymbolIndexExpr(val);
  else
    return QuestionmarkIndexExpr();
}

IndexExpr IndexExprBuilder::getIntArrayAsSymbol(
    Value intArrayVal, uint64_t i, int64_t defaultLiteral) {
  IndexExpr indexExpr = getIntArrayAsSymbol(intArrayVal, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LiteralIndexExpr(defaultLiteral) : indexExpr;
}

void IndexExprBuilder::getIntArrayAsSymbols(
    Value intArrayVal, IndexExprList &list, int64_t listSize) {
  list.clear();
  uint64_t size = getIntArraySize(intArrayVal);
  if (listSize == -1) // Meaning pick up the full size of the list.
    listSize = size;
  else
    assert((uint64_t)listSize <= size && "requesting too many elements");
  for (uint64_t i = 0; i < (uint64_t)listSize; ++i) {
    IndexExpr indexExpr = getIntArrayAsSymbol(intArrayVal, i);
    assert(!indexExpr.isUndefined() && "expected defined index expr");
    list.emplace_back(indexExpr);
  }
}

//===----------------------------------------------------------------------===//
// Get info from tensor/memref shape.

bool IndexExprBuilder::isLiteralShape(Value tensorOrMemrefValue, uint64_t i) {
  return getShape(tensorOrMemrefValue, i) != -1;
}

bool IndexExprBuilder::isLiteralShape(Value tensorOrMemrefValue) {
  uint64_t rank = getShapeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    if (!isLiteralShape(tensorOrMemrefValue, i))
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
  if (isLiteralShape(tensorOrMemrefValue, i))
    return getShapeAsLiteral(tensorOrMemrefValue, i);
  if (Value val = getShapeVal(tensorOrMemrefValue, i))
    return SymbolIndexExpr(val);
  return QuestionmarkIndexExpr(tensorOrMemrefValue, i);
}

IndexExpr IndexExprBuilder::getShapeAsDim(
    Value tensorOrMemrefValue, uint64_t i) {
  if (isLiteralShape(tensorOrMemrefValue, i))
    return getShapeAsLiteral(tensorOrMemrefValue, i);
  if (Value val = getShapeVal(tensorOrMemrefValue, i))
    return DimIndexExpr(val);
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
