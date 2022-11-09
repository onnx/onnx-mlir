/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToTOSACommon.hpp - ONNX dialects to TOSA lowering --------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//
namespace onnx_mlir {
namespace tosa {
// Lowers ReduceMean to a sequence of TOSA ops.
// Originates from the TorchToTosa conversion
llvm::Optional<mlir::Value> convertReduceMeanOp(mlir::PatternRewriter &rewriter,
    mlir::Operation *op, mlir::RankedTensorType output_type,
    mlir::Value input_value, mlir::ElementsAttr axes_elems, bool keep_dims);

} // namespace tosa
} // namespace onnx_mlir

namespace onnx_mlir {

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
void populateLoweringONNXSoftmaxOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXGemmOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXConvOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXReduceMeanOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
// `NN` directory methods:
void populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
    mlir::ConversionTarget &, mlir::RewritePatternSet &, mlir::TypeConverter &,
    mlir::MLIRContext *);
// `Tensor` directory methods:
void populateLoweringONNXReshapeOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXConstOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXPadOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
void populateLoweringONNXFlattenOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
// 'Flow' directory methods:
void populateLoweringONNXEntryPointOpToTOSAPattern(mlir::ConversionTarget &,
    mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
} // namespace onnx_mlir
