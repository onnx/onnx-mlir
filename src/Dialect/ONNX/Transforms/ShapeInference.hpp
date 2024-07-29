/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ShapeInference.hpp ---------------------------===//
//
// Shape inference patterns and helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SHAPE_INFERENCE_H
#define ONNX_MLIR_SHAPE_INFERENCE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Returns false if the rank or any dimension of any result types is unknown or
// dynamic.
bool returnsDynamicOrUnknownShape(mlir::Operation *op);

void getShapeInferencePatterns(mlir::RewritePatternSet &set);

// Propagates return op's operand types to f's return types.
// Works for both func::ReturnOp and ONNXReturnOp.
void inferFunctionReturnShapes(mlir::func::FuncOp f);

} // namespace onnx_mlir
#endif
