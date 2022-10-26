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
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H
#define ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H

#include "mlir/Dialect/Quant/QuantTypes.h" // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"   // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"                    // from @llvm-project

namespace mlir {
namespace tosa {

// Transpose a given TOSA Tensor
Value createTosaTransposedTensor(PatternRewriter &rewriter, Operation *op,
    Value &value, llvm::ArrayRef<int64_t> perm);

// Create a 32-bit float constant operator from a float
// The tensor will have the same rank as shape but with axis 1 (differs from
// tensorflow impl.)
Value getTosaConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
    float val, llvm::ArrayRef<int64_t> shape = {});

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
// To create INT48 TOSA constant, need to pass in llvm::APInt instead.
template <typename T>
llvm::Optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
    ArrayRef<T> vec, ArrayRef<int64_t> shape);

template <typename T>
T getValueFromTosaConst(Value &val) {
  return val.getDefiningOp<tosa::ConstOp>().getValue().cast<T>();
}

// Creates a TOSA operation and performs shape inference on the individual
// op. This allows shape inference during the framework to TOSA lowering.
template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInfer(
    PatternRewriter &rewriter, Location loc, Type result_ty, Args &&...args) {
  auto op = rewriter.create<TosaOp>(loc, result_ty, args...);

  InferShapedTypeOpInterface shapeInterface =
      dyn_cast<InferShapedTypeOpInterface>(op.getOperation());
  if (!shapeInterface)
    return op;

  SmallVector<ShapedTypeComponents> returnedShapes;
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
  auto result = op->getResult(0);
  auto predictedShape = returnedShapes[0];
  auto currentKnowledge = ValueKnowledge::getKnowledgeFromType(result_ty);

  // Compute the knowledge based on the inferred type.
  auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
  inferredKnowledge.dtype = result_ty.cast<ShapedType>().getElementType();
  inferredKnowledge.hasRank = predictedShape.hasRank();
  if (predictedShape.hasRank()) {
    for (auto dim : predictedShape.getDims()) {
      inferredKnowledge.sizes.push_back(dim);
    }
  }

  // Compute the new type based on the joined version.
  auto newKnowledge = ValueKnowledge::join(currentKnowledge, inferredKnowledge);
  auto new_ty = newKnowledge.getType();
  result.setType(new_ty);
  return op;
}

template <typename TosaOp, typename... Args>
void CreateReplaceOpAndInfer(
    PatternRewriter &rewriter, Operation *op, Type result_ty, Args &&...args) {
  auto result =
      CreateOpAndInfer<TosaOp>(rewriter, op->getLoc(), result_ty, args...);
  rewriter.replaceOp(op, result->getResults());
}

// Create a TOSA rescale op from input framework scaling, zero points and
// rounding mode
Value buildRescale(PatternRewriter &rewriter, Operation *op,
    ShapedType output_type, Value input_val, double scale, int64_t input_zp,
    int64_t output_zp, bool double_round, bool scale32);

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter &rewriter, Operation *op,
    Value input_val, double input_scale, int64_t input_zp);

} // namespace tosa
} // namespace mlir

#endif // ONNXMLIR_CONVERSION_ONNXTOTOSA_TOSALEGALIZEUTILS_H