/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==== ONNXToTosaLegalizeUtils.hpp - ONNX dialects to TOSA lowering Utils-===//
//
// Copyright 2020-2024 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common utils shared by the functions performing the
// lowering to the TOSA dialect. It is also used by TensorFlow and torch-mlir.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H
#define ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H

#include "mlir/Dialect/Quant/IR/QuantTypes.h" // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"     // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"   // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"                    // from @llvm-project
#include <src/Dialect/Mlir/IndexExpr.hpp>
#include <src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp>

namespace onnx_mlir {
namespace tosa {

// ONNX can use negative indices for axis while TOSA cannot. This functions
// makes sure the axis is in the right range for TOSA.
int64_t convertNegativeAxis(int64_t axis, int64_t inputRank);

// Get a vector of indexExpr and extract the Int64 values
llvm::SmallVector<int64_t> createInt64VectorFromIndexExpr(
    llvm::ArrayRef<IndexExpr> indexVector);

// Create a RankedTensorType with the given rank and all dims being 1
mlir::RankedTensorType reduceAxisToOne(
    int64_t rank, mlir::Type elementType, mlir::Attribute encoding = {});

// Returns the value TOSA ConstOp
template <typename T>
T getValueFromTosaConst(mlir::Value &val) {
  return mlir::cast<T>(val.getDefiningOp<mlir::tosa::ConstOp>().getValue());
}

// Retrieves an ElementsAttr out of a const operator.
// This function is made to work with both onnx.const and tosa.const
mlir::ElementsAttr getElementsAttrFromConst(mlir::Value &val);

// Takes a 1-d `tensor` with k elements and reshapes it into an `rank`-d or
// scalar tensor with shape {1, ..., 1, k, 1, ..., 1 } where `k` it at position
// `axis`.
mlir::Value expandShape(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::Value tensor, size_t axis, size_t rank);

// Creates a TOSA operation and performs shape inference on the individual
// op. This allows shape inference during the framework to TOSA lowering.
template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInfer(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::Type result_ty, Args &&... args) {

  auto op = rewriter.create<TosaOp>(loc, result_ty, args...);

  mlir::InferShapedTypeOpInterface shapeInterface =
      mlir::dyn_cast<mlir::InferShapedTypeOpInterface>(op.getOperation());
  if (!shapeInterface)
    return op;

  llvm::SmallVector<mlir::ShapedTypeComponents> returnedShapes;
  if (shapeInterface
          .inferReturnTypeComponents(op.getContext(), op.getLoc(),
              op->getOperands(), op->getAttrDictionary(),
              op->getPropertiesStorage(), op->getRegions(), returnedShapes)
          .failed())
    return op;

  // We need to use the element type of the existing result type to generate
  // the new result shaped type. This is because rescale can include a cast to
  // different bit-width types and does not have a TypeAttr to define the
  // target type.
  assert(returnedShapes.size() >= 1 && "Expected at least one returned shape");
  auto predictedShape = returnedShapes[0];
  if (predictedShape.hasRank())
    updateType(nullptr, op, predictedShape.getDims(),
        mlir::cast<mlir::ShapedType>(result_ty).getElementType());
  return op;
}

template <typename TosaOp, typename... Args>
void CreateReplaceOpAndInfer(mlir::PatternRewriter &rewriter,
    mlir::Operation *op, mlir::Type result_ty, Args &&... args) {
  auto result =
      CreateOpAndInfer<TosaOp>(rewriter, op->getLoc(), result_ty, args...);
  rewriter.replaceOp(op, result->getResults());
}

// Create a TOSA rescale op from input framework scaling, zero points and
// rounding mode
mlir::Value buildRescale(mlir::PatternRewriter &rewriter, mlir::Operation *op,
    mlir::ShapedType output_type, mlir::Value input_val, double scale,
    int64_t input_zp, int64_t output_zp, bool double_round, bool scale32);

// Creates TOSA rescale op with int32 output
mlir::Value buildRescaleToInt32(mlir::PatternRewriter &rewriter,
    mlir::Operation *op, mlir::Value input_val, double input_scale,
    int64_t input_zp);

/// Create a padding tosa::ConstOp from ONNX to Tosa format.
/// The two formats are:
/// ONNX : [b1, b2, b3, b4, e1, e2, e3, e4]
/// TOSA :[[b1, e1], [b2, e2], [b3, e3], [b4, e4]]
mlir::Value buildOnnxToTosaPaddingConstOp(mlir::PatternRewriter &rewriter,
    llvm::ArrayRef<int64_t> onnxPads, mlir::Location loc,
    const std::initializer_list<int64_t> &initialVals = {},
    const std::initializer_list<int64_t> &lastVals = {});

} // namespace tosa
} // namespace onnx_mlir

#endif // ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H
