/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- DevicePlacementHeuristic.hpp - Place ops using model  -------===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains heuristics to place operations on CPU or NNPA.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_HEURISTICS_H
#define ONNX_MLIR_HEURISTICS_H

#include "mlir/IR/BuiltinOps.h"

#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

namespace onnx_mlir {

using OpSetType = mlir::DenseSet<mlir::Operation *>;

/**
 *  Place all ops that qualify for NNPA executions on the NNPA.
 *
 * @param context Context of the model.
 * @param ops ONNX ops that should be considered in the device assignment.
 * @param cpuOps Set of ops that must execute on CPU.
 */

void PlaceAllLegalOpsOnNNPA(mlir::MLIRContext *context,
    const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const OpSetType &cpuOps);

/**
 *  Place all ops that qualify for NNPA execution on the NNPA when the
 * operations are estimated run faster on the NNPA.
 *
 * @param context Context of the model.
 * @param ops ONNX ops that should be considered in the device assignment.
 * @param dimAnalysis Pointer to dimension analysis tool for disambiguating
 * dynamic shape dimensions.
 * @param cpuOps Set of ops that must execute on CPU.
 */

void PlaceBeneficialOpsOnNNPA(mlir::MLIRContext *context,
    const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const DimAnalysis *dimAnalysis, const OpSetType &cpuOps);

/**
 * Place all operations that qualify for NNPA execution on the NNPA when the
 * operations are estimated run faster on the NNPA, including the costs of Stick
 * and Unstick necessary for NNPA execution. The algorithm starts to place on
 * the CPU/NNPA operations that are significantly faster on CPU/NNPA. Then it
 * aims to add operations to the NNPA when the new operations are faster
 * including the additional (if any) stick/unstick required for these less
 * significantly faster NNPA operations. Three factors below can modify the
 * sensitivity at which ops are assigned to the NNPA.
 *
 * @param context Context of the model.
 * @param ops ONNX ops that should be considered in the device assignment.
 * @param dimAnalysis Pointer to dimension analysis tool for disambiguating
 * dynamic shape dimensions.
 * @param cpuOps Set of ops that must execute on CPU.
 * @param minFactor NNPA (including stick/unstick) has to be at least minFactor
 * times faster than CPU for an op to be assigned to the NNPA.
 * @param significantCPUFactor CPU has to be at least significantFactor faster
 * than NNPA to seed/force computations on the CPU.
 * @param significantNNPAFactor NNPA has to be at least significantFactor faster
 * than CPU to seed/force computations on the NNPA.
 *
 * @note The significantCPUFactor can be smaller, as if it's not looking good
 * for the NNPA, we may as well seed the computation on CPU for ops that are
 * much better on the CPU. For significantNNPAFactor, we may want it much higher
 * as we might want only to send there really beneficial ops on the NNPA.
 * Combining a high significantNNPAFactor with a large minFactor, the heuristic
 * will put only ops that are really beneficial on the NNPA.
 */
void PlaceBeneficialOpsOnNNPAWithStickUnstick(mlir::MLIRContext *context,
    mlir::ModuleOp module, const llvm::SmallVector<mlir::Operation *, 32> &ops,
    const DimAnalysis *dimAnalysis, const OpSetType &cpuOps,
    double minFactor = 1.1, double significantCPUFactor = 2.0,
    double significantNNPAFactor = 3.0);

} // namespace onnx_mlir
#endif
