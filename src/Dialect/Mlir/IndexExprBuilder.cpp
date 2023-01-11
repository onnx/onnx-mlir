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
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Local helper.

namespace {

// Get scalar value regardless of the type.
// Code adapted from src/Dialect/ONNX/ONNXOps/OpHelper.cpp file.
template <typename RESULT_TYPE>
RESULT_TYPE getScalarValue(ElementsAttr &elementsAttr, Type type, uint64_t i) {
  Type elementaryType = getElementTypeOrSelf(type);
  if (elementaryType.isInteger(16) || elementaryType.isInteger(32) ||
      elementaryType.isInteger(64)) {
    auto value = elementsAttr.getValues<IntegerAttr>()[ArrayRef<uint64_t>({i})];
    return (RESULT_TYPE)value.cast<IntegerAttr>().getInt();
  } else if (elementaryType.isF32()) {
    auto value = elementsAttr.getValues<APFloat>()[ArrayRef<uint64_t>({i})];
    return (RESULT_TYPE)value.convertToFloat();
  } else if (elementaryType.isF64()) {
    auto value = elementsAttr.getValues<APFloat>()[ArrayRef<uint64_t>({i})];
    return (RESULT_TYPE)value.convertToDouble();
  }
  llvm_unreachable("Unexpected type.");
  return 0;
}

// Template instantiation for getScalarValue. I don't see any need to have any
// other result types that int, but keep it general just in case.
template int64_t getScalarValue<int64_t>(
    ElementsAttr &elementsAttr, Type type, uint64_t i);

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// IndexExprBuilder
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test/assert that value has type with defined shape and rank.

bool IndexExprBuilder::hasShapeAndRank(Value value) {
  ShapedType shapedType = value.getType().dyn_cast_or_null<ShapedType>();
  return shapedType && shapedType.hasRank();
}

void IndexExprBuilder::assertHasShapeAndRank(Value value) {
  assert(hasShapeAndRank(value) && "expected value with shape and rank");
}

//===----------------------------------------------------------------------===//
// Get Rank of Type / Size of 1D array

uint64_t IndexExprBuilder::getShapedTypeRank(Value value) {
  assertHasShapeAndRank(value);
  // Find shaped type size (rank of 0 is scalar).
  return value.getType().cast<ShapedType>().getRank();
}

// Size from 1D attribute array.
uint64_t IndexExprBuilder::getArraySize(ArrayAttr attrArray) {
  // Assume that if we have no array, a good value to return is 0.
  if (!attrArray)
    return 0;
  return attrArray.size();
}

// Size from 1D value array.
uint64_t IndexExprBuilder::getArraySize(Value array) {
  uint64_t rank = getShapedTypeRank(array);
  assert(rank < 2 && "expected a scalar or a 1 dimension array of int values");
  if (rank == 0)
    return 1;
  ShapedType shapeType = array.getType().cast<ShapedType>();
  return shapeType.getShape()[0];
}

//===----------------------------------------------------------------------===//
// Get literals from integer array attribute.

IndexExpr IndexExprBuilder::getIntFromArrayAsLiteral(
    ArrayAttr intAttrArray, uint64_t i) {
  uint64_t size = getArraySize(intAttrArray);
  if (i >= size)
    return UndefinedIndexExpr();
  int64_t val = (intAttrArray.getValue()[i]).cast<IntegerAttr>().getInt();
  return LiteralIndexExpr(val);
}

IndexExpr IndexExprBuilder::getIntFromArrayAsLiteral(
    ArrayAttr intAttrArray, uint64_t i, int64_t outOfBoundVal) {
  IndexExpr indexExpr = getIntFromArrayAsLiteral(intAttrArray, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LiteralIndexExpr(outOfBoundVal) : indexExpr;
}

void IndexExprBuilder::getIntFromArrayAsLiterals(
    ArrayAttr intAttrArray, IndexExprList &list, int64_t len) {
  list.clear();
  uint64_t size = getArraySize(intAttrArray);
  if (len == -1) // Meaning pick up the full size of the list.
    len = size;
  else
    assert((uint64_t)len <= size && "requesting too many elements");
  for (uint64_t i = 0; i < (uint64_t)len; ++i) {
    IndexExpr indexExpr = getIntFromArrayAsLiteral(intAttrArray, i);
    assert(!indexExpr.isUndefined() && "expected defined index expr");
    list.emplace_back(indexExpr);
  }
}

void IndexExprBuilder::getIntFromArrayAsLiterals(ArrayAttr intAttrArray,
    int64_t outOfBoundVal, IndexExprList &list, int64_t len) {
  list.clear();
  assert(len >= 0 && "expect a defined size");
  for (uint64_t i = 0; i < (uint64_t)len; ++i) {
    IndexExpr indexExpr =
        getIntFromArrayAsLiteral(intAttrArray, i, outOfBoundVal);
    assert(!indexExpr.isUndefined() && "expected defined index expr");
    list.emplace_back(indexExpr);
  }
}

//===----------------------------------------------------------------------===//
// Get symbols from value defined by intVal.

//===----------------------------------------------------------------------===//
// Get Index Expr from value defined by array.

// Function that perform the work, creating Literal, Symbol, or Dim IndexExpr.
IndexExpr IndexExprBuilder::getIntFromArray(
    Value array, uint64_t i, bool makeSymbol) {
  uint64_t size = getArraySize(array);
  Type type = array.getType();

  if (i >= size)
    return UndefinedIndexExpr();
  if (ElementsAttr elementsAttr = getConst(array)) {
    // From OpHelper.cpp's getScalarValue.
    int64_t intVal = getScalarValue<int64_t>(elementsAttr, type, i);
    return LiteralIndexExpr(intVal);
  }
  // If our scalar array is not a constant; we have a runtime value.
  if (Value val = getVal(array, i)) {
    // Assume that we can write code.
    MathBuilder createMath(*this);
    Value castedVal = createMath.cast(b().getIndexType(), val);
    if (makeSymbol)
      return SymbolIndexExpr(castedVal);
    else
      return DimIndexExpr(castedVal);
  } else {
    return QuestionmarkIndexExpr();
  }
}

IndexExpr IndexExprBuilder::getIntAsSymbol(Value value) {
  assert(getArraySize(value) == 1 && "Expected a scalar");
  return getIntFromArrayAsSymbol(value, 0);
}

IndexExpr IndexExprBuilder::getIntAsDim(Value value) {
  assert(getArraySize(value) == 1 && "Expected a scalar");
  return getIntFromArrayAsDim(value, 0);
}

IndexExpr IndexExprBuilder::getIntFromArrayAsSymbol(Value array, uint64_t i) {
  return getIntFromArray(array, i, /*makeSymbol*/ true);
}

IndexExpr IndexExprBuilder::getIntFromArrayAsDim(Value array, uint64_t i) {
  return getIntFromArray(array, i, /*makeSymbol*/ false);
}

IndexExpr IndexExprBuilder::getIntFromArrayAsSymbol(
    Value array, uint64_t i, int64_t defaultLiteral) {
  IndexExpr indexExpr = getIntFromArrayAsSymbol(array, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LiteralIndexExpr(defaultLiteral) : indexExpr;
}

IndexExpr IndexExprBuilder::getIntFromArrayAsDim(
    Value array, uint64_t i, int64_t defaultLiteral) {
  IndexExpr indexExpr = getIntFromArrayAsDim(array, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LiteralIndexExpr(defaultLiteral) : indexExpr;
}

void IndexExprBuilder::getIntFromArrayAsSymbols(
    Value array, IndexExprList &list, int64_t len) {
  list.clear();
  uint64_t size = getArraySize(array);
  if (len == -1) // Meaning pick up the full size of the list.
    len = size;
  else
    assert((uint64_t)len <= size && "requesting too many elements");
  for (uint64_t i = 0; i < (uint64_t)len; ++i) {
    IndexExpr indexExpr = getIntFromArrayAsSymbol(array, i);
    assert(!indexExpr.isUndefined() && "expected defined index expr");
    list.emplace_back(indexExpr);
  }
}

void IndexExprBuilder::getIntFromArrayAsDims(
    Value array, IndexExprList &list, int64_t len) {
  list.clear();
  uint64_t size = getArraySize(array);
  if (len == -1) // Meaning pick up the full size of the list.
    len = size;
  else
    assert((uint64_t)len <= size && "requesting too many elements");
  for (uint64_t i = 0; i < (uint64_t)len; ++i) {
    IndexExpr indexExpr = getIntFromArrayAsDim(array, i);
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
  uint64_t rank = getShapedTypeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    if (!isLiteralShape(tensorOrMemrefValue, i))
      return false;
  return true;
}

int64_t IndexExprBuilder::getShape(Value tensorOrMemrefValue, uint64_t i) {
  uint64_t rank = getShapedTypeRank(tensorOrMemrefValue);
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
  uint64_t rank = getShapedTypeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    list.emplace_back(getShapeAsLiteral(tensorOrMemrefValue, i));
}

void IndexExprBuilder::getShapeAsSymbols(
    Value tensorOrMemrefValue, IndexExprList &list) {
  list.clear();
  uint64_t rank = getShapedTypeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    list.emplace_back(getShapeAsSymbol(tensorOrMemrefValue, i));
}

void IndexExprBuilder::getShapeAsDims(
    Value tensorOrMemrefValue, IndexExprList &list) {
  list.clear();
  uint64_t rank = getShapedTypeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    list.emplace_back(getShapeAsDim(tensorOrMemrefValue, i));
}

} // namespace onnx_mlir
