/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- NNPAPlacementReporter.hpp -----------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Instrumentation pass that reports ONNX operations running on CPU.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NNPA_PLACEMENT_REPORTER_H
#define ONNX_MLIR_NNPA_PLACEMENT_REPORTER_H

#include "mlir/Pass/PassInstrumentation.h"

namespace mlir {
class MLIRContext;
}

namespace onnx_mlir {
namespace zhigh {

struct NNPAPlacementReporter : public mlir::PassInstrumentation {
  NNPAPlacementReporter(mlir::MLIRContext *context);
  ~NNPAPlacementReporter() override;

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;
};

} // namespace zhigh
} // namespace onnx_mlir
#endif