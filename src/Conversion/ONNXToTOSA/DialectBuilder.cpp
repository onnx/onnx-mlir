
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ DialectBuilder.hpp - TOSA dialect builder --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains the dialect build for the TOSA dialect. Uses the same
// implementation as ONNXToMhlo with minor differences.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include <src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp>

using namespace mlir;

namespace onnx_mlir {

template <typename T>
bool testNumberOfElementsMatch(ArrayRef<T> vec, ArrayRef<int64_t> shape) {
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

Value TosaBuilder::getConst(ArrayRef<int64_t> vec, ArrayRef<int64_t> shape) {

  assert(testNumberOfElementsMatch(vec, shape) &&
         "getConstTensor(): number of elements mismatch.");

  auto constType = RankedTensorType::get(
      shape, rewriter().getIntegerType(sizeof(int64_t) * 8));

  Value constOp = this->createConstFromRankedTensorAndVec(vec, constType);
  return constOp;
}

Value TosaBuilder::getConst(ArrayRef<int32_t> vec, ArrayRef<int64_t> shape) {

  assert(testNumberOfElementsMatch(vec, shape) &&
         "getConstTensor(): number of elements mismatch.");

  auto constType = RankedTensorType::get(
      shape, rewriter().getIntegerType(sizeof(int32_t) * 8));

  Value constOp = this->createConstFromRankedTensorAndVec(vec, constType);
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
  assert(testNumberOfElementsMatch(vec, shape) &&
         "getConstTensor(): number of elements mismatch.");

  auto constType = RankedTensorType::get(shape, rewriter().getF32Type());

  Value constOp = this->createConstFromRankedTensorAndVec(vec, constType);
  return constOp;
}

Value TosaBuilder::getConst(float val, llvm::ArrayRef<int64_t> shape) {
  auto constType = tosa::reduceAxisToOne(shape, rewriter().getF32Type());
  auto constAttr = DenseElementsAttr::get(constType, val);

  auto constOp =
      rewriter().create<mlir::tosa::ConstOp>(loc(), constType, constAttr);
  return constOp;
}

Value TosaBuilder::reshape(mlir::Value &value, llvm::ArrayRef<int64_t> shape) {
  ArrayAttr shapeAttr = rewriter().getI64ArrayAttr(shape);
  auto valueType = value.getType().cast<ShapedType>();
  Type newValueType =
      RankedTensorType::get(llvm::SmallVector<int64_t, 4>(shape.size(), -1),
          valueType.getElementType());
  return tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(
      rewriter(), loc(), newValueType, value, shapeAttr);
}

Value TosaBuilder::transpose(mlir::Value &value, llvm::ArrayRef<int32_t> perm) {
  // Create Permutation Const
  Value permList = this->getConst(
      perm, {value.getType().cast<RankedTensorType>().getRank()});
  auto valueType = value.getType().cast<ShapedType>();
  // get new value type
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(valueType.getShape().size(), -1),
      valueType.getElementType());
  // create transpose for value
  Value newValue = tosa::CreateOpAndInfer<mlir::tosa::TransposeOp>(
      rewriter(), loc(), newValueType, value, permList);
  return newValue;
}

Value TosaBuilder::slice(Value &inputConst, llvm::ArrayRef<int64_t> size,
    llvm::ArrayRef<int64_t> start) {
  ArrayAttr sizeAttr = rewriter().getI64ArrayAttr(size);
  ArrayAttr startAttr = rewriter().getI64ArrayAttr(start);
  Value newSliceInput =
      tosa::CreateOpAndInfer<mlir::tosa::SliceOp>(rewriter(), loc(),
          RankedTensorType::get(llvm::SmallVector<int64_t, 4>(size.size(), -1),
              inputConst.getType().cast<ShapedType>().getElementType()),
          inputConst, startAttr, sizeAttr);
  return newSliceInput;
}

llvm::Optional<Value> TosaBuilder::gather(Value resultValue, Value inputValue,
    Value indicesValue, int32_t batchDims, int32_t axis) {
  return tosa::convertGatherOp(rewriter(), loc(), resultValue, inputValue,
      indicesValue, batchDims, axis);
};

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
    if (constOp.value().has_value())
      return constOp.valueAttr().dyn_cast<DenseElementsAttr>();
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
