/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==== ONNXToTosaLegalizeUtils.cpp - ONNX dialects to TOSA lowering Utils-===//
//
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common utils shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"   // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"

#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp" // from @llvm-project
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include <cstdint>
#include <src/Dialect/Mlir/IndexExpr.hpp>

using namespace mlir;
namespace onnx_mlir {
namespace tosa {

llvm::SmallVector<int64_t> createInt64VectorFromIndexExpr(
    llvm::ArrayRef<IndexExpr> indexVector) {
  llvm::SmallVector<int64_t, 4> literalVector;
  for (const auto &indexExpr : indexVector) {
    literalVector.push_back(indexExpr.getLiteral());
  }
  return literalVector;
}

Value sliceTensor(PatternRewriter &rewriter, Operation *op, Value &inputConst,
    const llvm::ArrayRef<int64_t> &size, const llvm::ArrayRef<int64_t> &start) {
  ArrayAttr sizeAttr = rewriter.getI64ArrayAttr(size);
  ArrayAttr startAttr = rewriter.getI64ArrayAttr(start);
  Value newSliceInput =
      tosa::CreateOpAndInfer<mlir::tosa::SliceOp>(rewriter, op->getLoc(),
          RankedTensorType::get(llvm::SmallVector<int64_t, 4>(size.size(), -1),
              inputConst.getType().cast<ShapedType>().getElementType()),
          inputConst, startAttr, sizeAttr);
  return newSliceInput;
}

Value createTosaTransposedTensor(PatternRewriter &rewriter, Operation *op,
    Value &value, llvm::ArrayRef<int64_t> perm) {
  // Create Permutation Const
  Value permList = getConstTensor(
      rewriter, op, perm, {value.getType().cast<RankedTensorType>().getRank()})
                       .value();
  auto valueType = value.getType().cast<ShapedType>();
  // get new value type
  Type newValueType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(valueType.getShape().size(), -1),
      valueType.getElementType());
  // create transpose for value
  Value newValue = CreateOpAndInfer<mlir::tosa::TransposeOp>(
      rewriter, op->getLoc(), newValueType, value, permList);
  return newValue;
}

mlir::RankedTensorType reduceAxisToOne(llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType, mlir::Attribute encoding = {}) {
  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(shape.size(), 1), elementType, encoding);
}

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
    float val, llvm::ArrayRef<int64_t> shape) {
  auto constType = reduceAxisToOne(shape, rewriter.getF32Type());
  auto constAttr = DenseElementsAttr::get(constType, val);

  auto constOp =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), constType, constAttr);
  return constOp.getResult();
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
llvm::Optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
    ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op = rewriter.create<mlir::tosa::ConstOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for float
template <>
llvm::Optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
    Operation *op, ArrayRef<float> vec, ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op = rewriter.create<mlir::tosa::ConstOp>(
      op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template instantiation
template llvm::Optional<Value> getConstTensor<int32_t>(PatternRewriter &,
    Operation *, ArrayRef<int32_t> vec, ArrayRef<int64_t> shape);

template llvm::Optional<Value> getConstTensor<int64_t>(PatternRewriter &,
    Operation *, ArrayRef<int64_t> vec, ArrayRef<int64_t> shape);

// Create a TOSA rescale op from input framework tensor, zero points and
// rounding mode
Value buildRescale(PatternRewriter &rewriter, Operation *op,
    ShapedType output_type, Value input_val, double scale, int64_t input_zp,
    int64_t output_zp, bool double_round, bool scale32) {
  int32_t multiplier;
  int32_t shift;

  int32_t scale_width = scale32 ? 32 : 16;

  mlir::tosa::computeMultiplierAndShift(scale, multiplier, shift, scale_width);

  auto rescale_op = CreateOpAndInfer<mlir::tosa::RescaleOp>(rewriter,
      op->getLoc(), output_type, input_val,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(input_zp)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(output_zp)),
      rewriter.getI32ArrayAttr({multiplier}), rewriter.getI32ArrayAttr({shift}),
      rewriter.getBoolAttr(scale32), rewriter.getBoolAttr(double_round),
      rewriter.getBoolAttr(false));

  return rescale_op.getResult();
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter &rewriter, Operation *op,
    Value input_val, double input_scale, int64_t input_zp) {
  // Output is always int32 type
  auto input_type = input_val.getType().dyn_cast<mlir::ShapedType>();
  assert(input_type);
  auto output_type = input_type.clone(rewriter.getI32Type());

  return buildRescale(rewriter, op, output_type, input_val, input_scale,
      input_zp, 0, false, true);
}

} // namespace tosa
} // namespace onnx_mlir
