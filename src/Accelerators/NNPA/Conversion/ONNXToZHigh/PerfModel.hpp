/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- PerfModel.hpp - Estimate if CPU or NNPA is faster  ----------===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains methods that estimates the computational time of an ONNX
// operation on a CPU and NNPA for a z16 and indicates which device is faster.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PERF_H
#define ONNX_MLIR_PERF_H

#include "mlir/IR/BuiltinOps.h"

#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

// When an op has a model, define the CPU and NNPA estimated times and return
// true. When an op does not have a model, just return false.
bool estimateTimeForOpWithModel(mlir::Operation *op,
    const DimAnalysis *dimAnalysis, double &cpuEstimatedTime,
    double &nnpaEstimatedTime);

// Estimate the CPU time for stick/unstick given the shape in oper
double estimateTimeForStickOp(mlir::Value oper);
double estimateTimeForUnstickOp(mlir::Value oper);

} // namespace onnx_mlir
#endif
