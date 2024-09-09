/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ DialectBuilder.hpp - TOSA dialect builder --------------------===//
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains the dialect build for the TOSA dialect. Uses the same
// implementation as ONNXToStablehlo with minor differences.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename T>
bool TosaBuilder::testNumberOfElementsMatch(
    ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  uint64_t numTotalElements = 1;
  for (int64_t a : shape) {
    numTotalElements *= a;
  }
  return (vec.size() == numTotalElements);
}

template <typename T>
Value TosaBuilder::createConstFromRankedTensorAndVec(
    ArrayRef<T> vec, RankedTensorType &constType) {
  auto constAttr = DenseElementsAttr::get(constType, vec);

  Value constOp =
      rewriter().create<mlir::tosa::ConstOp>(loc(), constType, constAttr);
  return constOp;
}

template <typename T>
Value TosaBuilder::createConst(
    ArrayRef<T> vec, ArrayRef<int64_t> shape, Type &type) {
  assert(testNumberOfElementsMatch(vec, shape) &&
         "getConstTensor(): number of elements mismatch.");

  auto constType = RankedTensorType::get(shape, type);

  Value constOp = this->createConstFromRankedTensorAndVec(vec, constType);
  return constOp;
}

bool TosaBuilder::needsRankBroadcast(ValueRange valueRange) {
  int64_t firstRank = mlir::cast<ShapedType>(valueRange[0].getType()).getRank();
  for (Value operand : valueRange) {
    auto operandType = mlir::cast<ShapedType>(operand.getType());
    if (firstRank != operandType.getRank())
      return true;
  }
  return false;
}

Value TosaBuilder::expandRank(Value input, int64_t rank) {
  auto inputType = mlir::cast<ShapedType>(input.getType());
  int64_t inputRank = inputType.getRank();
  assert(inputRank <= rank && "cannot reduce rank of operation");
  if (inputRank == rank)
    return input;

  llvm::SmallVector<int64_t, 4> newShape(rank - inputRank, 1);
  llvm::transform(inputType.getShape(), std::back_inserter(newShape),
      [](const int64_t shape) { return shape; });
  return this->reshape(input, newShape);
}

llvm::SmallVector<Value, 4> TosaBuilder::equalizeRanks(ValueRange valueRange) {
  // Get highest rank from the operands.
  int64_t maxRank = 0;
  for (auto type : valueRange.getTypes()) {
    int64_t currentRank = mlir::cast<ShapedType>(type).getRank();
    maxRank = std::max(maxRank, currentRank);
  }
  llvm::SmallVector<Value, 4> reshapedValues;
  // Iterate through all values comparing the rank.
  for (auto value : valueRange) {
    auto shapedType = mlir::cast<ShapedType>(value.getType());
    int64_t currentRank = shapedType.getRank();
    // Only add a reshape op if necessary.
    if (maxRank > currentRank) {
      reshapedValues.push_back(this->expandRank(value, maxRank));
      continue;
    }
    reshapedValues.push_back(value);
  }
  return reshapedValues;
}

Value TosaBuilder::getConst(ArrayRef<int64_t> vec, ArrayRef<int64_t> shape) {
  auto elementType = rewriter().getIntegerType(sizeof(int64_t) * 8);
  Value constOp = this->createConst<int64_t>(vec, shape, elementType);
  return constOp;
}

Value TosaBuilder::getConst(ArrayRef<int32_t> vec, ArrayRef<int64_t> shape) {
  auto elementType = rewriter().getIntegerType(sizeof(int32_t) * 8);
  Value constOp = this->createConst<int32_t>(vec, shape, elementType);
  return constOp;
}

Value TosaBuilder::getConst(ArrayRef<float> vec, ArrayRef<int64_t> shape) {
  auto elementType = rewriter().getF32Type();
  Value constOp = this->createConst<float>(vec, shape, elementType);
  return constOp;
}

Value TosaBuilder::getSplattedConst(float val, llvm::ArrayRef<int64_t> shape) {
  auto constType = tosa::reduceAxisToOne(shape, rewriter().getF32Type());
  auto constAttr = DenseElementsAttr::get(constType, val);

  auto constOp =
      rewriter().create<mlir::tosa::ConstOp>(loc(), constType, constAttr);
  return constOp;
}

Value TosaBuilder::transpose(Value &value, llvm::ArrayRef<int32_t> perm) {
  int64_t valueRank = mlir::cast<RankedTensorType>(value.getType()).getRank();
  assert((valueRank == static_cast<int64_t>(perm.size())) &&
         "value and perm vector don't have the same rank");
  // Create Permutation Const
  Value permList = this->getConst(perm, {valueRank});
  auto valueType = mlir::cast<ShapedType>(value.getType());
  // get new value type
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(
          valueType.getShape().size(), ShapedType::kDynamic),
      valueType.getElementType());
  // create transpose for value
  Value newValue = tosa::CreateOpAndInfer<mlir::tosa::TransposeOp>(
      rewriter(), loc(), newValueType, value, permList);
  return newValue;
}

Value TosaBuilder::slice(Value &inputConst, llvm::ArrayRef<int64_t> size,
    llvm::ArrayRef<int64_t> start) {
  DenseI64ArrayAttr sizeAttr = rewriter().getDenseI64ArrayAttr(size);
  DenseI64ArrayAttr startAttr = rewriter().getDenseI64ArrayAttr(start);
  Value newSliceInput =
      tosa::CreateOpAndInfer<mlir::tosa::SliceOp>(rewriter(), loc(),
          RankedTensorType::get(
              llvm::SmallVector<int64_t, 4>(size.size(), ShapedType::kDynamic),
              mlir::cast<ShapedType>(inputConst.getType()).getElementType()),
          inputConst, startAttr, sizeAttr);
  return newSliceInput;
}

Value TosaBuilder::reshape(Value &value, llvm::ArrayRef<int64_t> shape) {
  auto shapeAttr = rewriter().getDenseI64ArrayAttr(shape);
  auto valueType = mlir::cast<ShapedType>(value.getType());
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(shape.size(), ShapedType::kDynamic),
      valueType.getElementType());
  return tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(
      rewriter(), loc(), newValueType, value, shapeAttr);
}

Value TosaBuilder::mul(Value &lhs, Value &rhs, int32_t shift) {
  if (needsRankBroadcast({lhs, rhs})) {
    llvm::SmallVector<Value, 4> valueVec = equalizeRanks({lhs, rhs});
    lhs = valueVec[0];
    rhs = valueVec[1];
  }
  auto lhsType = mlir::cast<ShapedType>(lhs.getType());
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(lhsType.getRank(), ShapedType::kDynamic),
      lhsType.getElementType());
  return tosa::CreateOpAndInfer<mlir::tosa::MulOp>(
      rewriter(), loc(), newValueType, lhs, rhs, shift);
}

Value TosaBuilder::intdiv(Value &lhs, Value &rhs) {
  Type lhsElementType = mlir::cast<ShapedType>(lhs.getType()).getElementType();
  Type rhsElementType = mlir::cast<ShapedType>(rhs.getType()).getElementType();
  assert((lhsElementType.isSignlessInteger(32) &&
             rhsElementType.isSignlessInteger(32)) &&
         "Tosa IntDivOp needs 32-bit signless integer inputs");

  if (needsRankBroadcast({lhs, rhs})) {
    llvm::SmallVector<Value, 4> valueVec = equalizeRanks({lhs, rhs});
    lhs = valueVec[0];
    rhs = valueVec[1];
  }

  auto lhsType = mlir::cast<ShapedType>(lhs.getType());
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(lhsType.getRank(), ShapedType::kDynamic),
      lhsElementType);
  return tosa::CreateOpAndInfer<mlir::tosa::IntDivOp>(
      rewriter(), loc(), newValueType, lhs, rhs);
}

Value TosaBuilder::reciprocal(Value &input) {
  auto inputType = mlir::cast<ShapedType>(input.getType());
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(inputType.getRank(), ShapedType::kDynamic),
      inputType.getElementType());
  return tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(
      rewriter(), loc(), newValueType, input);
}

template <typename T>
Value TosaBuilder::binaryOp(Value &lhs, Value &rhs) {
  if (needsRankBroadcast({lhs, rhs})) {
    llvm::SmallVector<Value, 4> valueVec = equalizeRanks({lhs, rhs});
    lhs = valueVec[0];
    rhs = valueVec[1];
  }
  auto lhsType = mlir::cast<ShapedType>(lhs.getType());
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(lhsType.getRank(), ShapedType::kDynamic),
      lhsType.getElementType());
  return tosa::CreateOpAndInfer<T>(rewriter(), loc(), newValueType, lhs, rhs);
}

template Value TosaBuilder::binaryOp<mlir::tosa::AddOp>(Value &lhs, Value &rhs);

template Value TosaBuilder::binaryOp<mlir::tosa::SubOp>(Value &lhs, Value &rhs);
// =============================================================================
// IndexExpr Builder for Lowering using Shape/TOSA Dialect.
// =============================================================================

// Return null if none is found.
ElementsAttr IndexExprBuilderForTosa::getConst(Value value) {
  auto definingOp = value.getDefiningOp();
  // If we have a cast between index/integer, skip it, i.e. get the defining op
  // that is the input to the cast.
  if (auto castOp = dyn_cast_or_null<arith::IndexCastOp>(definingOp)) {
    Value input = castOp.getIn();
    definingOp = input.getDefiningOp();
  }
  if (auto constOp = dyn_cast_or_null<mlir::tosa::ConstOp>(definingOp)) {
    if (constOp.getValueAttr())
      return mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  } else if (auto constOp = dyn_cast_or_null<ONNXConstantOp>(definingOp)) {
    if (constOp.getValue().has_value())
      return mlir::dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  }
  return nullptr;
}

Value IndexExprBuilderForTosa::getVal(Value intArrayVal, uint64_t i) {
  MultiDialectBuilder<AffineBuilder, MathBuilder> create(*this);
  // Need to add some acceptable dialects to TOSA conversion.
  llvm_unreachable(
      "unimplemented (see IndexExprBuilderForKrnl for functionality).");
}

Value IndexExprBuilderForTosa::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  ShapeBuilder createShape(*this);
  return createShape.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
