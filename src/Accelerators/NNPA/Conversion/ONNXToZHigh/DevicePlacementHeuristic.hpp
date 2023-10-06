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

// minFactor: NNPA has to be at least minFactor times faster than CPU.
//
// significantCPUFactor: CPU has to be at least significantFactor faster than
// NNPA to seed/force computations on the CPU.
//
// significantNNPAFactor: NNPA has to be at least significantFactor faster than
// CPU to seed/force computations on the NNPA.
//
// CPU factor can be smaller, as if it's not looking good for the NNPA, we may
// as well seed the computation on CPU for ops that are much better on the CPU.
// For NNPA factor, we may want it much higher as we might want only to send
// there really beneficial ops on the NNPA. Combining a high NNPA factor with a
// large minFactor, the heuristic will put only ops that are really beneficial
// on the NNPA.

void PlaceBeneficialOpsOnNNPAWithStickUnstick(mlir::MLIRContext *context,
    mlir::ModuleOp module, const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const DimAnalysis *dimAnalysis, const OpSetType &cpuOps,
    double minFactor = 1.1, double significantCPUFactor = 2.0,
    double significantNNPAFactor = 3.0);

} // namespace onnx_mlir
