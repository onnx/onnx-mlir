/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToTOSACommon.hpp - ONNX dialects to TOSA lowering --------===//
//
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "DialectBuilder.hpp"
#include "ONNXToTOSALegalizeUtils.hpp"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
namespace tosa {

// Common function for lowering reduce operations to TOSA ops.
// Modified from TensorFlow
template <typename T>
llvm::Optional<mlir::Value> convertReduceOpCommon(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    mlir::RankedTensorType outputType, mlir::Value inputValue,
    mlir::ElementsAttr axesElems, bool keepDims, mlir::Type reduceElementType) {
  TosaBuilder tosaBuilder(rewriter, op->getLoc());
  mlir::RankedTensorType inputType =
      inputValue.getType().dyn_cast<mlir::RankedTensorType>();
  if (!inputType)
    return std::nullopt;

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  auto inputRank = inputShape.size();

  if (axesElems.getNumElements() == 0) {
    // No axes means return the original tensor.
    auto identityOp = onnx_mlir::tosa::CreateOpAndInfer<mlir::tosa::IdentityOp>(
        rewriter, op->getLoc(), outputType, inputValue);
    return identityOp.getResult();
  }
  // Reduce along each axis
  llvm::SmallVector<int64_t> shapeVec(inputShape.begin(), inputShape.end());
  mlir::Value newValue = inputValue;
  for (int i = 0; i < axesElems.getNumElements(); i++) {
    int64_t axisVal = axesElems.getValues<mlir::IntegerAttr>()[i].getInt();
    if (axisVal < 0)
      axisVal += inputRank;
    auto axisAttr = rewriter.getI64IntegerAttr(axisVal);

    shapeVec[axisVal] = 1;
    mlir::RankedTensorType reduceType =
        mlir::RankedTensorType::get(shapeVec, reduceElementType);

    auto reduceOp = CreateOpAndInfer<T>(
        rewriter, op->getLoc(), reduceType, newValue, axisAttr);

    newValue = reduceOp.getResult();
  }

  // Optionally squeeze out the reduced axes.
  if (!keepDims) {
    newValue = tosaBuilder.reshape(newValue, outputShape);
  }
  return newValue;
}

} // namespace tosa

//===----------------------------------------------------------------------===//
// Check for valid TOSA types.
//===----------------------------------------------------------------------===//

inline bool isTOSASignedInt(mlir::Type type) {
  mlir::IntegerType intType = type.dyn_cast<mlir::IntegerType>();
  std::set<unsigned> intWidth{8, 16, 32, 48, 64};
  return intType && intType.isSignless() &&
         (intWidth.find(intType.getWidth()) != intWidth.end());
}

inline bool isTOSAFloat(mlir::Type type) {
  return type.isa<mlir::BFloat16Type, mlir::Float16Type, mlir::Float32Type>();
}

//===----------------------------------------------------------------------===//
// This is to get a TOSA operation of a given type for a specific operation.
//===----------------------------------------------------------------------===//
template <typename ONNXOp>
struct TOSADialectOp {
  using Op = void;
};

template <typename Op>
using TOSAOp = typename TOSADialectOp<Op>::Op;

// `Math` directory methods:
void populateLoweringONNXElementwiseOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXGemmOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXSoftmaxOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXReduceMeanOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXConvOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
// `NN` directory methods:
void populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
    mlir::ConversionTarget &, mlir::RewritePatternSet &, mlir::TypeConverter &,
    mlir::MLIRContext *);
// `Tensor` directory methods:
void populateLoweringONNXConstOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXReshapeOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
} // namespace onnx_mlir
