/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToTOSACommon.hpp - ONNX dialects to TOSA lowering --------===//
//
// Copyright 2020-2024 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_TO_TOSA_H
#define ONNX_MLIR_ONNX_TO_TOSA_H

#include "DialectBuilder.hpp"
#include "ONNXToTOSALegalizeUtils.hpp"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
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
std::optional<mlir::Value> convertReduceOpCommon(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    mlir::RankedTensorType outputType, mlir::Value inputValue,
    mlir::ElementsAttr axesElems, bool keepDims, mlir::Type reduceElementType);

// This calculates the values that need to be added to the padding in order to
// simulate the ceil mode
template <typename ShapeHelperType>
llvm::SmallVector<int64_t> getCeilConstants(llvm::ArrayRef<int64_t> inputShape,
    ONNXGenericPoolOpShapeHelper<ShapeHelperType> &shapeHelper,
    int64_t ceilMode);

// Create an ArrayAttr of pad from \p shapeHelper using \p padIndexOrder.
// Values are calculated considering \p ceilMode.
template <typename ShapeHelperType>
mlir::ArrayAttr createOrderedPadAttrForWindowBasedOps(
    mlir::PatternRewriter &rewriter, const llvm::ArrayRef<int64_t> inputShape,
    ONNXGenericPoolOpShapeHelper<ShapeHelperType> &shapeHelper,
    const int64_t ceilMode, const llvm::ArrayRef<int64_t> padIndexOrder);

inline mlir::LogicalResult getAvgPool2dAccType(mlir::PatternRewriter &rewriter,
    mlir::Value input, mlir::TypeAttr &accType);

// Lower MaxPool and AveragePool to TOSA ops.
template <typename ONNXPoolOp, typename TOSAPoolOp>
mlir::FailureOr<mlir::Value> convertPoolOp(
    mlir::PatternRewriter &rewriter, mlir::Operation *op);

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp.inc"

} // namespace tosa

//===----------------------------------------------------------------------===//
// Check for valid TOSA types.
//===----------------------------------------------------------------------===//

inline bool isTOSASignedInt(mlir::Type type) {
  mlir::IntegerType intType = mlir::dyn_cast<mlir::IntegerType>(type);
  std::set<unsigned> intWidth{1, 8, 16, 32, 48, 64};
  return intType && intType.isSignless() &&
         (intWidth.find(intType.getWidth()) != intWidth.end());
}

inline bool isTOSAFloat(mlir::Type type) {
  return mlir::isa<mlir::BFloat16Type, mlir::Float16Type, mlir::Float32Type>(
      type);
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
void populateLoweringONNXAveragePoolOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
// `Tensor` directory methods:
void populateLoweringONNXConstOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXReshapeOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXResizeOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
} // namespace onnx_mlir
#endif
