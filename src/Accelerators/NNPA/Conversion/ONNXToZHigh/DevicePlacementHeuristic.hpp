/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- DevicePlacementHeuristic.hpp - Place ops using model  -------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains heuristics to place operations on CPU or NNPA.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"

#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

using OpSetType = mlir::DenseSet<mlir::Operation *>;

void PlaceAllLegalOpsOnNNPA(mlir::MLIRContext *context,
    const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const OpSetType &cpuOps);

void PlaceBeneficialOpsOnNNPA(mlir::MLIRContext *context,
    const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const DimAnalysis *dimAnalysis, const OpSetType &cpuOps);

void PlaceBeneficialOpsOnNNPAWithStickUnstick(mlir::MLIRContext *context,
    mlir::ModuleOp module, const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const DimAnalysis *dimAnalysis, const OpSetType &cpuOps);

} // namespace onnx_mlir
