/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==== ONNXToTosaLegalizeUtils.hpp - ONNX dialects to TOSA lowering Utils-===//
//
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common utils shared by the functions performing the
// lowering to the TOSA dialect. It is also used by TensorFlow and torch-mlir.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H
#define ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H

#include "mlir/Dialect/Quant/QuantTypes.h"        // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"         // from @llvm-project
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

// Slices a TOSA Tensor with the specific size and start values
mlir::Value sliceTensor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
    mlir::Value &inputConst, const llvm::ArrayRef<int64_t> &size,
    const llvm::ArrayRef<int64_t> &start);

// Transpose a given TOSA Tensor
mlir::Value createTosaTransposedTensor(mlir::PatternRewriter &rewriter,
    mlir::Operation *op, mlir::Value &value, llvm::ArrayRef<int64_t> perm);

// Create a 32-bit float constant operator from a float
// The tensor will have the same rank as shape but with axis 1 (differs from
// tensorflow impl.)
mlir::Value getTosaConstTensorSingleF32(mlir::PatternRewriter &rewriter,
    mlir::Operation *op, float val, llvm::ArrayRef<int64_t> shape = {});

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
// To create INT48 TOSA constant, need to pass in llvm::APInt instead.
template <typename T>
llvm::Optional<mlir::Value> getConstTensor(mlir::PatternRewriter &rewriter,
    mlir::Operation *op, llvm::ArrayRef<T> vec, llvm::ArrayRef<int64_t> shape);

template <typename T>
T getValueFromTosaConst(mlir::Value &val) {
  return val.getDefiningOp<mlir::tosa::ConstOp>().getValue().cast<T>();
}

// Creates a TOSA operation and performs shape inference on the individual
// op. This allows shape inference during the framework to TOSA lowering.
template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInfer(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::Type result_ty, Args &&... args) {
  auto op = rewriter.create<TosaOp>(loc, result_ty, args...);

  mlir::InferShapedTypeOpInterface shapeInterface =
      llvm::dyn_cast<mlir::InferShapedTypeOpInterface>(op.getOperation());
  if (!shapeInterface)
    return op;

  llvm::SmallVector<mlir::ShapedTypeComponents> returnedShapes;
  if (shapeInterface
          .inferReturnTypeComponents(op.getContext(), op.getLoc(),
              op->getOperands(), op->getAttrDictionary(), op->getRegions(),
              returnedShapes)
          .failed())
    return op;

  // We need to use the element type of the existing result type to generate
  // the new result shaped type. This is because rescale can include a cast to
  // different bit-width types and does not have a TypeAttr to define the
  // target type.
  auto predictedShape = returnedShapes[0];
  if (predictedShape.hasRank())
    updateType(op, predictedShape.getDims(),
        result_ty.cast<mlir::ShapedType>().getElementType());
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

} // namespace tosa
} // namespace onnx_mlir

#endif // ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H