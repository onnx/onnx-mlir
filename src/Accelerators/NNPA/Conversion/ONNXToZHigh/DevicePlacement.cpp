/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- DevicePlacement.cpp - Device Placement for NNPA -------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This pass is to set device (CPU, or NNPA) for each operation in ONNX level.
// Device placement can be decided by:
// - user configuration file if given
// - a cost model
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "device-placement"

using namespace mlir;

namespace {

struct DevicePlacementPass
    : public PassWrapper<DevicePlacementPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DevicePlacementPass)

  StringRef getArgument() const override { return "device-placement"; }

  StringRef getDescription() const override {
    return "Device placement for NNPA";
  }

  void runOnOperation() final;
};

void DevicePlacementPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a DevicePlacement pass.
 */
std::unique_ptr<mlir::Pass> createDevicePlacementPass() {
  return std::make_unique<DevicePlacementPass>();
}

} // namespace onnx_mlir
