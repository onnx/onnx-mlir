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

static bool hasShapeAndRank(Value val) {
  ShapedType shapedType = val.getType().dyn_cast_or_null<ShapedType>();
  return shapedType && shapedType.hasRank();
}

// Get scalar value regardless of the type.
// Code adapted from Dialect/ONNX/ONNXOps/OpHelper.cpp file.
template <typename RESULT_TYPE>
static RESULT_TYPE getScalarValue(
    DenseElementsAttr &denseAttr, Type type, uint64_t i) {
  Type elementaryType = getElementTypeOrSelf(type);
  ArrayRef<uint64_t> index({i});
  if (elementaryType.isInteger(16) || elementaryType.isInteger(32) ||
      elementaryType.isInteger(64)) {
    auto value = denseAttr.getValues<IntegerAttr>()[index];
    return (RESULT_TYPE)value.cast<IntegerAttr>().getInt();
  } else if (elementaryType.isF32()) {
    auto value = denseAttr.getValues<APFloat>()[index];
    return (RESULT_TYPE)value.convertToFloat();
  } else if (elementaryType.isF64()) {
    auto value = denseAttr.getValues<APFloat>()[index];
    return (RESULT_TYPE)value.convertToDouble();
  }
  llvm_unreachable("Unexpected type.");
  return 0;
}

// Template instantiation for getScalarValue. I don't see any need to have any
// other result types that int, but keep it general just in case.
template static int64_t getScalarValue<int64_t>(
    DenseElementsAttr &denseAttr, Type type, uint64_t i);

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// IndexShapeBuilder
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Get Rank of Type / Size of 1D array

uint64_t IndexExprBuilder::getTypeRank(Value value) {
  assert(hasShapeAndRank(value) && "expected shaped type with rank");
  // Find shaped type size (rank of 0 is scalar).
  return value.getType().cast<ShapedType>().getRank();
}

// Size from 1D attribute array.
uint64_t IndexExprBuilder::getArraySize(ArrayAttr attrArray) {
  return attrArray.size();
}

// Size from 1D value array.
uint64_t IndexExprBuilder::getArraySize(Value array) {
  uint64_t rank = getTypeRank(array);
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
  uint64_t size = intAttrArray.size();
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

//===----------------------------------------------------------------------===//
// Get symbols from value defined by intVal.

IndexExpr IndexExprBuilder::getIntAsSymbol(Value value) {
  assert(getArraySize(value) == 1 && "Expected a scalar");
  return getIntArrayAsSymbol(value, 0);
}

//===----------------------------------------------------------------------===//
// Get symbols from value defined by array.

IndexExpr IndexExprBuilder::getIntArrayAsSymbol(Value array, uint64_t i) {
  uint64_t size = getArraySize(array);
  Type type = array.getType();

  if (i >= size)
    return UndefinedIndexExpr();
  if (DenseElementsAttr denseAttr = getConst(array)) {
    // From OpHelper.cpp's getScalarValue.
    int64_t intVal = getScalarValue<int64_t>(denseAttr, type, i);
    return LiteralIndexExpr(intVal);
  }
  // If our scalar array is not a constant; we have a questionmark.
  if (Value val = getVal(array, i)) {
    // Assume that we can write code.
    MathBuilder createMath(*this);
    Value intVal = createMath.cast(b().getIndexType(), val);
    return SymbolIndexExpr(intVal);
  }
  else
    return QuestionmarkIndexExpr();
}

IndexExpr IndexExprBuilder::getIntArrayAsSymbol(
    Value array, uint64_t i, int64_t defaultLiteral) {
  IndexExpr indexExpr = getIntArrayAsSymbol(array, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LiteralIndexExpr(defaultLiteral) : indexExpr;
}

void IndexExprBuilder::getIntArrayAsSymbols(
    Value array, IndexExprList &list, int64_t len) {
  list.clear();
  uint64_t size = getArraySize(array);
  if (len == -1) // Meaning pick up the full size of the list.
    len = size;
  else
    assert((uint64_t)len <= size && "requesting too many elements");
  for (uint64_t i = 0; i < (uint64_t)len; ++i) {
    IndexExpr indexExpr = getIntArrayAsSymbol(array, i);
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
  uint64_t rank = getTypeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    if (!isLiteralShape(tensorOrMemrefValue, i))
      return false;
  return true;
}

int64_t IndexExprBuilder::getShape(Value tensorOrMemrefValue, uint64_t i) {
  uint64_t rank = getTypeRank(tensorOrMemrefValue);
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
  uint64_t rank = getTypeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    list.emplace_back(getShapeAsLiteral(tensorOrMemrefValue, i));
}

void IndexExprBuilder::getShapeAsSymbols(
    Value tensorOrMemrefValue, IndexExprList &list) {
  list.clear();
  uint64_t rank = getTypeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    list.emplace_back(getShapeAsSymbol(tensorOrMemrefValue, i));
}

void IndexExprBuilder::getShapeAsDims(
    Value tensorOrMemrefValue, IndexExprList &list) {
  list.clear();
  uint64_t rank = getTypeRank(tensorOrMemrefValue);
  for (uint64_t i = 0; i < rank; ++i)
    list.emplace_back(getShapeAsDim(tensorOrMemrefValue, i));
}

} // namespace onnx_mlir
