/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ DialectBuilder.hpp - TOSA dialect builder --------------------===//
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains the dialect build for the TOSA dialect. Uses the same
// implementation as ONNXToMhlo with minor differences.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
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
  int64_t firstRank = valueRange[0].getType().cast<ShapedType>().getRank();
  for (Value operand : valueRange) {
    auto operandType = operand.getType().cast<ShapedType>();
    if (firstRank != operandType.getRank())
      return true;
  }
  return false;
}

Value TosaBuilder::expandRank(Value input, int64_t rank) {
  auto inputType = input.getType().cast<ShapedType>();
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
    int64_t currentRank = type.cast<ShapedType>().getRank();
    maxRank = std::max(maxRank, currentRank);
  }
  llvm::SmallVector<Value, 4> reshapedValues;
  // Iterate through all values comparing the rank.
  for (auto value : valueRange) {
    auto shapedType = value.getType().cast<ShapedType>();
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

Value TosaBuilder::getConst(ArrayRef<int8_t> vec, ArrayRef<int64_t> shape) {
  assert(testNumberOfElementsMatch(vec, shape) &&
         "getConstTensor(): number of elements mismatch.");

  auto constType = RankedTensorType::get(shape, rewriter().getI8Type());

  Value constOp = this->createConstFromRankedTensorAndVec(vec, constType);
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

Value TosaBuilder::transpose(mlir::Value &value, llvm::ArrayRef<int32_t> perm) {
  int64_t valueRank = value.getType().cast<RankedTensorType>().getRank();
  assert((valueRank == (int64_t)perm.size()) &&
         "value and perm vector don't have the same rank");
  // Create Permutation Const
  Value permList = this->getConst(perm, {valueRank});
  auto valueType = value.getType().cast<ShapedType>();
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
              inputConst.getType().cast<ShapedType>().getElementType()),
          inputConst, startAttr, sizeAttr);
  return newSliceInput;
}

std::optional<Value> TosaBuilder::gather(Value resultValue, Value inputValue,
    Value indicesValue, int32_t batchDims, int32_t axis) {
  return tosa::convertGatherOp(rewriter(), loc(), resultValue, inputValue,
      indicesValue, batchDims, axis);
};
Value TosaBuilder::reshape(mlir::Value &value, llvm::ArrayRef<int64_t> shape) {
  auto shapeAttr = rewriter().getDenseI64ArrayAttr(shape);
  auto valueType = value.getType().cast<ShapedType>();
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(shape.size(), ShapedType::kDynamic),
      valueType.getElementType());
  return tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(
      rewriter(), loc(), newValueType, value, shapeAttr);
}

Value TosaBuilder::mul(mlir::Value &lhs, mlir::Value &rhs, int32_t shift) {
  if (needsRankBroadcast({lhs, rhs})) {
    llvm::SmallVector<Value, 4> valueVec = equalizeRanks({lhs, rhs});
    lhs = valueVec[0];
    rhs = valueVec[1];
  }
  auto lhsType = lhs.getType().cast<ShapedType>();
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(lhsType.getRank(), ShapedType::kDynamic),
      lhsType.getElementType());
  return tosa::CreateOpAndInfer<mlir::tosa::MulOp>(
      rewriter(), loc(), newValueType, lhs, rhs, shift);
}

Value TosaBuilder::intdiv(mlir::Value &lhs, mlir::Value &rhs) {
  Type lhsElementType = lhs.getType().cast<ShapedType>().getElementType();
  Type rhsElementType = rhs.getType().cast<ShapedType>().getElementType();
  assert((lhsElementType.isSignlessInteger(32) &&
             rhsElementType.isSignlessInteger(32)) &&
         "Tosa DivOp needs 32-bit signless integer inputs");

  if (needsRankBroadcast({lhs, rhs})) {
    llvm::SmallVector<Value, 4> valueVec = equalizeRanks({lhs, rhs});
    lhs = valueVec[0];
    rhs = valueVec[1];
  }

  auto lhsType = lhs.getType().cast<ShapedType>();
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(lhsType.getRank(), ShapedType::kDynamic),
      lhsElementType);
  return tosa::CreateOpAndInfer<mlir::tosa::DivOp>(
      rewriter(), loc(), newValueType, lhs, rhs);
}

Value TosaBuilder::reciprocal(mlir::Value &input) {
  auto inputType = input.getType().cast<ShapedType>();
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(inputType.getRank(), ShapedType::kDynamic),
      inputType.getElementType());
  return tosa::CreateOpAndInfer<mlir::tosa::ReciprocalOp>(
      rewriter(), loc(), newValueType, input);
}

template <typename T>
Value TosaBuilder::binaryOp(mlir::Value &lhs, mlir::Value &rhs) {
  if (needsRankBroadcast({lhs, rhs})) {
    llvm::SmallVector<Value, 4> valueVec = equalizeRanks({lhs, rhs});
    lhs = valueVec[0];
    rhs = valueVec[1];
  }
  auto lhsType = lhs.getType().cast<ShapedType>();
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(lhsType.getRank(), ShapedType::kDynamic),
      lhsType.getElementType());
  return tosa::CreateOpAndInfer<T>(rewriter(), loc(), newValueType, lhs, rhs);
}

template Value TosaBuilder::binaryOp<mlir::tosa::AddOp>(
    mlir::Value &lhs, mlir::Value &rhs);

template Value TosaBuilder::binaryOp<mlir::tosa::SubOp>(
    mlir::Value &lhs, mlir::Value &rhs);

static bool containsNonZero(llvm::SmallVectorImpl<int64_t> &values) {
  return llvm::any_of(values, [](int64_t value) { return value != 0; });
}

FailureOr<Value> TosaBuilder::resizeWindowBasedOps(mlir::Value &value,
    const llvm::ArrayRef<int64_t> inputShape,
    const llvm::ArrayRef<int64_t> weightSpatialShape,
    llvm::SmallVectorImpl<int64_t> &padding,
    const llvm::ArrayRef<int64_t> strides,
    const llvm::ArrayRef<int64_t> dilation) {

  // Returns the number of unused values at the end of a dimension
  auto getOffset = [](int64_t inputDimension, int64_t outputDimension,
                       int64_t kernelDimension, int64_t padFront,
                       int64_t padBack, int64_t stride, int64_t dilation) {
    int64_t offset = inputDimension + padFront + padBack -
                     dilation * (kernelDimension - 1) - 1 -
                     outputDimension * stride + stride;
    assert(offset >= 0);
    return offset;
  };

  auto getOutputSpatialDimension =
      [](int64_t inputDimension, int64_t kernelDimension, int64_t padFront,
          int64_t padBack, int64_t stride, int64_t dilation) {
        int64_t outputSpatialDimension =
            std::floor((inputDimension + padFront + padBack -
                        dilation * (kernelDimension - 1) - 1)) /
                stride +
            1;
        return outputSpatialDimension;
      };

  // Only the end of a dimension is cut or padded differently. The beginning
  // is unchanged.
  llvm::SmallVector<int64_t, 2> cellsToCut;
  llvm::SmallVector<int64_t, 2> cellsToPad;
  for (int i = 0; i < 2; i++) {
    int64_t padFront = padding[2 * i];
    int64_t padBack = padding[2 * i + 1];
    int64_t outputSpatialDimension =
        getOutputSpatialDimension(inputShape[i + 1], weightSpatialShape[i],
            padFront, padBack, strides[i], dilation[i]);
    int64_t offset = getOffset(inputShape[i + 1], outputSpatialDimension,
        weightSpatialShape[i], padFront, padBack, strides[i], dilation[i]);
    if (offset > padBack) {
      cellsToPad.push_back(0);
      cellsToCut.push_back(offset - padBack);
    } else {
      cellsToPad.push_back(padBack - offset);
      cellsToCut.push_back(0);
    }
  }

  // Edge case where the kernel only uses padding values and none of the actual
  // input values
  if ((inputShape[1] - cellsToCut[0] == 0) ||
      (inputShape[2] - cellsToCut[1] == 0))
    return rewriter().notifyMatchFailure(
        loc(), "the operation does not use any value of the input tensor");

  // Only slice if we actually need it
  if (containsNonZero(cellsToCut)) {
    value = this->slice(value,
        {inputShape[0], inputShape[1] - cellsToCut[0],
            inputShape[2] - cellsToCut[1], inputShape[3]},
        {0, 0, 0, 0});
  }
  padding[1] = cellsToPad[0];
  padding[3] = cellsToPad[1];

  return value;
}

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
      return constOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  } else if (auto constOp = dyn_cast_or_null<ONNXConstantOp>(definingOp)) {
    if (constOp.getValue().has_value())
      return constOp.getValueAttr().dyn_cast<DenseElementsAttr>();
  }
  return nullptr;
}

Value IndexExprBuilderForTosa::getVal(Value intArrayVal, uint64_t i) {
  MultiDialectBuilder<AffineBuilder, MathBuilder> create(*this);
  // Need to add some acceptable dialects to MHLO conversion.
  llvm_unreachable(
      "unimplemented (see IndexExprBuilderForKrnl for functionality).");
}

Value IndexExprBuilderForTosa::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  ShapeBuilder createShape(*this);
  return createShape.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
