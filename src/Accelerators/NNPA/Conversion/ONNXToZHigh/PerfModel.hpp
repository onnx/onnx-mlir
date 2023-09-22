/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- PerfModel.hpp - Estimate if CPU or NNPA is faster  ----------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains methods that estimates the computational time of an ONNX
// operation on a CPU and NNPA for a z16 and indicates which device is faster.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

bool isOpFasterOnNNPA(mlir::Operation *op, const DimAnalysis *dimAnalysis);

}
