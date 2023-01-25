
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

using namespace mlir;

namespace onnx_mlir {

template <typename T>
Value TosaBuilder::getConst(ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  assert((vec.size() == num_total_elements) &&
         "getConstTensor(): number of elements mismatch.");

  auto const_type =
      RankedTensorType::get(shape, rewriter().getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter().create<mlir::tosa::ConstOp>(loc(), const_type, const_attr);
  return const_op;
}

// Template specialization for float
template <>
Value TosaBuilder::getConst<float>(
    ArrayRef<float> vec, ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  assert((vec.size() == num_total_elements) &&
         "getConstTensor(): number of elements mismatch.");

  auto const_type = RankedTensorType::get(shape, rewriter().getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter().create<mlir::tosa::ConstOp>(loc(), const_type, const_attr);
  return const_op;
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

Value TosaBuilder::transpose(mlir::Value &value, llvm::ArrayRef<int64_t> perm) {
  // Create Permutation Const
  Value permList = this->getConst<int64_t>(
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

// =============================================================================
// IndexExpr Builder for Lowering using Shape/TOSA Dialect.
// =============================================================================

// Return null if none is found.
DenseElementsAttr IndexExprBuilderForTosa::getConst(Value value) {
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
