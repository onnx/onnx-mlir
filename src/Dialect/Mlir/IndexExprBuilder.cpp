/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ IndexExprBuilder.cpp - builder for index expressions ----===//
//
// Copyright 2022-2024 The IBM Research Authors.
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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"
#include "src/Support/Arrays.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Local helper.

namespace {

APFloat getFloatValue(ElementsAttr elementsAttr, Type elType, uint64_t i) {
  // Work around that elementsAttr may be a DenseUI8ResourceElementsAttr
  // which doesn't support getValues<APFloat>().
  if (auto resource = dyn_cast<DenseUI8ResourceElementsAttr>(elementsAttr)) {
    ArrayRef<uint8_t> array = resource.tryGetAsArrayRef().value();
    if (elType.isF32())
      return APFloat(onnx_mlir::castArrayRef<float>(array)[i]);
    if (elType.isF64())
      return APFloat(onnx_mlir::castArrayRef<double>(array)[i]);
    llvm_unreachable("Unexpected float type");
  }
  return elementsAttr.getValues<APFloat>()[i];
}

APInt getIntValue(ElementsAttr elementsAttr, Type elType, uint64_t i) {
  // Work around that elementsAttr may be a DenseUI8ResourceElementsAttr
  // which doesn't support getValues<APInt>().
  if (auto resource = dyn_cast<DenseUI8ResourceElementsAttr>(elementsAttr)) {
    ArrayRef<uint8_t> array = resource.tryGetAsArrayRef().value();
    bool isSigned = true;
    if (elType.isInteger(16))
      return APInt(16, onnx_mlir::castArrayRef<int16_t>(array)[i], isSigned);
    if (elType.isInteger(32))
      return APInt(32, onnx_mlir::castArrayRef<int32_t>(array)[i], isSigned);
    if (elType.isInteger(64))
      return APInt(64, onnx_mlir::castArrayRef<int64_t>(array)[i], isSigned);
    llvm_unreachable("Unexpected int type");
  }
  return elementsAttr.getValues<APInt>()[i];
}

} // namespace

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// IndexExprBuilder
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test/assert that value has type with defined shape and rank.

// Warning, this does not work well in presence of Seq and Opt types, which have
// a dependence on ONNX.
bool IndexExprBuilder::hasShapeAndRank(Value value) {
  assert(value && "expected a value");
  ShapedType shapedType = mlir::dyn_cast_or_null<ShapedType>(value.getType());
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
  return mlir::cast<ShapedType>(value.getType()).getRank();
}

// Size from 1D attribute array.
int64_t IndexExprBuilder::getArraySize(ArrayAttr attrArray) {
  // Assume that if we have no array, a good value to return is 0.
  if (!attrArray)
    return 0;
  return attrArray.size();
}

// Size from 1D value array.
int64_t IndexExprBuilder::getArraySize(Value array, bool staticSizeOnly) {
  uint64_t rank = getShapedTypeRank(array);
  assert(rank < 2 && "expected a scalar or a 1 dimension array of int values");
  if (rank == 0)
    return 1;
  int64_t shape = getShape(array, 0);
  if (staticSizeOnly)
    assert(shape != ShapedType::kDynamic && "expected static size");
  return shape;
}

//===----------------------------------------------------------------------===//
// Get literals from integer array attribute.

IndexExpr IndexExprBuilder::getIntFromArrayAsLiteral(
    ArrayAttr intAttrArray, uint64_t i) {
  uint64_t size = getArraySize(intAttrArray);
  if (i >= size)
    return UndefinedIndexExpr();
  int64_t val = mlir::cast<IntegerAttr>(intAttrArray.getValue()[i]).getInt();
  return LitIE(val);
}

IndexExpr IndexExprBuilder::getIntFromArrayAsLiteral(
    ArrayAttr intAttrArray, uint64_t i, int64_t outOfBoundVal) {
  IndexExpr indexExpr = getIntFromArrayAsLiteral(intAttrArray, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LitIE(outOfBoundVal) : indexExpr;
}

void IndexExprBuilder::getIntFromArrayAsLiterals(
    ArrayAttr intAttrArray, IndexExprList &list, int64_t len) {
  list.clear();
  uint64_t size = getArraySize(intAttrArray);
  if (len == -1) // Meaning pick up the full size of the list.
    len = size;
  else
    assert(
        static_cast<uint64_t>(len) <= size && "requesting too many elements");
  if (len == 0)
    return;
  for (uint64_t i = 0; i < static_cast<uint64_t>(len); ++i) {
    IndexExpr indexExpr = getIntFromArrayAsLiteral(intAttrArray, i);
    assert(!indexExpr.isUndefined() && "expected defined index expr");
    list.emplace_back(indexExpr);
  }
}

void IndexExprBuilder::getIntFromArrayAsLiterals(ArrayAttr intAttrArray,
    int64_t outOfBoundVal, IndexExprList &list, int64_t len) {
  list.clear();
  assert(len >= 0 && "expect a defined size");
  if (len == 0)
    return;
  for (uint64_t i = 0; i < static_cast<uint64_t>(len); ++i) {
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

// Function that perform the work, creating Literal (int/float), Symbol(int),
// Dim(int), NonAffine (float) IndexExpr.
IndexExpr IndexExprBuilder::getValFromArray(
    Value array, uint64_t i, bool makeSymbol, bool isFloat, int64_t arraySize) {
  Type elType = getElementTypeOrSelf(array);
  if (isFloat)
    assert(isa<FloatType>(elType) && "array element type mismatch");
  else
    assert(isa<IntegerType>(elType) && "array element type mismatch");
  int64_t size = getArraySize(array, /*static only*/ false);
  if (arraySize >= 0) {
    // Was given a defined arraySize value.
    if (size == ShapedType::kDynamic)
      // Could not derive an array size from value, use given arraySize.
      size = arraySize;
    else
      // Was able to derive an array size from array value.
      assert(arraySize == size && "expected given size to be the same as the "
                                  "one detected from the array value");
  }
  if (size == ShapedType::kDynamic || i >= static_cast<uint64_t>(size)) {
    return UndefinedIndexExpr();
  }
  if (ElementsAttr elementsAttr = getConst(array)) {
    if (isFloat) {
      double floatVal =
          getFloatValue(elementsAttr, elType, i).convertToDouble();
      return LitIE(floatVal);
    } else {
      int64_t intVal = getIntValue(elementsAttr, elType, i).getSExtValue();
      return LitIE(intVal);
    }
  }
  // If our scalar array is not a constant; we have a runtime value.
  // Sometimes, a specific value can be a constant. E.g. the array is defined by
  // a ONNXConcat that mixes constant and runtime values.
  if (Value val = getVal(array, i)) {
    // getVal may be smart enough to derive a constant value at index i.
    // In such a case, we can return a literal.
    if (ElementsAttr elementsAttr = getConst(val)) {
      if (isFloat) {
        double floatVal =
            getFloatValue(elementsAttr, elType, 0).convertToDouble();
        return LitIE(floatVal);
      } else {
        int64_t intVal = getIntValue(elementsAttr, elType, 0).getSExtValue();
        return LitIE(intVal);
      }
    }
    // Otherwise, we can write code.
    MathBuilder createMath(*this);
    if (isFloat) {
      Value castedVal = createMath.cast(b().getF32Type(), val);
      return NonAffineIndexExpr(castedVal);
    }
    Value castedVal = createMath.castToIndex(val);
    if (makeSymbol)
      return SymIE(castedVal);
    else
      return DimIE(castedVal);
  }
  return QuestionmarkIndexExpr(isFloat);
}

IndexExpr IndexExprBuilder::getIntAsSymbol(Value value) {
  assert(getArraySize(value, /*static only*/ true) == 1 && "Expected a scalar");
  return getIntFromArrayAsSymbol(value, 0);
}

IndexExpr IndexExprBuilder::getIntAsDim(Value value) {
  assert(getArraySize(value, /*static only*/ true) == 1 && "Expected a scalar");
  return getIntFromArrayAsDim(value, 0);
}

IndexExpr IndexExprBuilder::getFloatAsNonAffine(Value value) {
  assert(getArraySize(value, /*static only*/ true) == 1 && "Expected a scalar");
  return getFloatFromArrayAsNonAffine(value, 0);
}

IndexExpr IndexExprBuilder::getIntFromArrayAsSymbol(
    Value intArray, uint64_t i, int64_t arraySize) {
  return getValFromArray(
      intArray, i, /*makeSymbol*/ true, /*isFloat*/ false, arraySize);
}

IndexExpr IndexExprBuilder::getIntFromArrayAsDim(
    Value intArray, uint64_t i, int64_t arraySize) {
  return getValFromArray(
      intArray, i, /*makeSymbol*/ false, /*isFloat*/ false, arraySize);
}

IndexExpr IndexExprBuilder::getFloatFromArrayAsNonAffine(
    Value floatArray, uint64_t i, int64_t arraySize) {
  return getValFromArray(
      floatArray, i, /*makeSymbol*/ false, /*isFloat*/ true, arraySize);
}

IndexExpr IndexExprBuilder::getIntFromArrayAsSymbolWithOutOfBound(
    Value intArray, uint64_t i, int64_t defaultLiteral) {
  IndexExpr indexExpr = getIntFromArrayAsSymbol(intArray, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LitIE(defaultLiteral) : indexExpr;
}

IndexExpr IndexExprBuilder::getIntFromArrayAsDimWithOutOfBound(
    Value intArray, uint64_t i, int64_t defaultLiteral) {
  IndexExpr indexExpr = getIntFromArrayAsDim(intArray, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LitIE(defaultLiteral) : indexExpr;
}

IndexExpr IndexExprBuilder::getFloatFromArrayAsNonAffineWithOutOfBound(
    Value floatArray, uint64_t i, double defaultLiteral) {
  IndexExpr indexExpr = getFloatFromArrayAsNonAffine(floatArray, i);
  // Undefined value are set to default value.
  return indexExpr.isUndefined() ? LitIE(defaultLiteral) : indexExpr;
}

void IndexExprBuilder::getIntFromArrayAsSymbols(
    Value intArray, IndexExprList &list, int64_t len) {
  list.clear();
  uint64_t size = getArraySize(intArray, /*static only*/ true);
  if (len == -1) // Meaning pick up the full size of the list.
    len = size;
  else
    assert(
        static_cast<uint64_t>(len) <= size && "requesting too many elements");
  if (len == 0)
    return;
  for (uint64_t i = 0; i < static_cast<uint64_t>(len); ++i) {
    IndexExpr indexExpr = getIntFromArrayAsSymbol(intArray, i);
    assert(!indexExpr.isUndefined() && "expected defined index expr");
    list.emplace_back(indexExpr);
  }
}

void IndexExprBuilder::getIntFromArrayAsDims(
    Value intArray, IndexExprList &list, int64_t len) {
  list.clear();
  uint64_t size = getArraySize(intArray, /*static only*/ true);
  if (len == -1) // Meaning pick up the full size of the list.
    len = size;
  else
    assert(
        static_cast<uint64_t>(len) <= size && "requesting too many elements");
  if (len == 0)
    return;
  for (uint64_t i = 0; i < static_cast<uint64_t>(len); ++i) {
    IndexExpr indexExpr = getIntFromArrayAsDim(intArray, i);
    assert(!indexExpr.isUndefined() && "expected defined index expr");
    list.emplace_back(indexExpr);
  }
}

void IndexExprBuilder::getFloatFromArrayAsNonAffine(
    Value floatArray, IndexExprList &list, int64_t len) {
  list.clear();
  uint64_t size = getArraySize(floatArray, /*static only*/ true);
  if (len == -1) // Meaning pick up the full size of the list.
    len = size;
  else
    assert(
        static_cast<uint64_t>(len) <= size && "requesting too many elements");
  if (len == 0)
    return;
  for (uint64_t i = 0; i < static_cast<uint64_t>(len); ++i) {
    IndexExpr indexExpr = getFloatFromArrayAsNonAffine(floatArray, i);
    assert(!indexExpr.isUndefined() && "expected defined index expr");
    list.emplace_back(indexExpr);
  }
}

//===----------------------------------------------------------------------===//
// Get info from tensor/memref shape.

bool IndexExprBuilder::isLiteralShape(Value tensorOrMemrefValue, uint64_t i) {
  return getShape(tensorOrMemrefValue, i) != ShapedType::kDynamic;
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
  return mlir::cast<ShapedType>(tensorOrMemrefValue.getType()).getShape()[i];
}

// Get index expressions from tensor/memref shape.
IndexExpr IndexExprBuilder::getShapeAsLiteral(
    Value tensorOrMemrefValue, uint64_t i) {
  int64_t shape = getShape(tensorOrMemrefValue, i);
  assert(
      shape != ShapedType::kDynamic && "expected compile time constant shape");
  return LitIE(shape);
}

IndexExpr IndexExprBuilder::getShapeAsSymbol(
    Value tensorOrMemrefValue, uint64_t i) {
  if (isLiteralShape(tensorOrMemrefValue, i))
    return getShapeAsLiteral(tensorOrMemrefValue, i);
  if (Value val = getShapeVal(tensorOrMemrefValue, i))
    return SymIE(val);
  return QuestionmarkIndexExpr(tensorOrMemrefValue, i);
}

IndexExpr IndexExprBuilder::getShapeAsDim(
    Value tensorOrMemrefValue, uint64_t i) {
  if (isLiteralShape(tensorOrMemrefValue, i))
    return getShapeAsLiteral(tensorOrMemrefValue, i);
  if (Value val = getShapeVal(tensorOrMemrefValue, i))
    return DimIE(val);
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

//===----------------------------------------------------------------------===//
// Support tiles.

// Determined if we have a full tile (affine expression compared to >=0)
IndexExpr IndexExprBuilder::isTileFull(
    IndexExpr i, IndexExpr block, IndexExpr UB) {
  // Determine if the current tile is full. It is full if the beginning of
  // the tile (i) is smaller or equal to UB - bloc, namely
  //   PredicateIndexExpr nIsFullTile = (i<= (nUB - nBlock));
  // However, if UB is divisible by Block, then its full no matter what.
  if (UB.isLiteral() && (UB.getLiteral() % block.getLiteral() == 0)) {
    // Last tile is guaranteed to be full because UB is divisible by block.
    return LitIE(1); // 1 >= 0 is true
  }
  // True if i <= (UB - block), namely UB - block - i >= 0.
  // Affine expressions compared to >= 0
  IndexExpr res = UB - block;
  res = res - i;
  return res;
}

// If tile is not full, the remaining trip count within the tile.
IndexExpr IndexExprBuilder::partialTileSize(
    IndexExpr i, IndexExpr block, IndexExpr UB) {
  // Trip count for partial tiles: leftover = UB - i in general. If UB is
  // known at compile time, then without loss of generality, leftover = (UB-
  // i) % Block, and since i is by definition a multiple of Block (i is
  // index at beginning of tile), then leftover = UB % Block.
  if (UB.isLiteral()) {
    IndexExpr partialTrip = UB % block;
    assert(partialTrip.isLiteral() && "op on 2 literals has to be literal");
    return partialTrip;
  }
  // don't have to take the mod since we know we have a partial tile already.
  return UB - i;
}

// Regardless of if the tile is full or partial, the trip count withint the
// tile.
IndexExpr IndexExprBuilder::tileSize(
    IndexExpr i, IndexExpr block, IndexExpr UB) {
  // Trip count in general: min(UB - i, Block).
  // UB.debugPrint("trip UB");
  // block.debugPrint("trip block");
  // i.debugPrint("trip GI");
  if (UB.isLiteral() && (UB.getLiteral() % block.getLiteral() == 0)) {
    // Last tile is guaranteed to be full, so trip is always full.
    return block;
  }
  return IndexExpr::min(UB - i, block);
}

} // namespace onnx_mlir
