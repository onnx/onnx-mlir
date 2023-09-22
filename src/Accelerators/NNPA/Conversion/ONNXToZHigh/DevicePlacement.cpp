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
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "device-placement"

using namespace mlir;
using namespace onnx_mlir;

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
  using OpSetType = DenseSet<Operation *>;
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  // Run the unknown dimension analysis to help check equality of unknown
  // dimensions at compile time.
  DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

  // Cost model and user configuration file go here if it's given.
  // (Reserved for cost model and user configuration file)
  bool useCostBenefitEstimation = true;

  // Run patterns that converts ONNX to ZHigh with analysis mode to collect
  // operations that are not converted. Those non-converted ops are running on
  // the host instead of accelerator.
  // Keep the order of calling pass synced with RewriteONNXForZHigh.cpp and
  // ONNXToZHigh.cpp.

  OpSetType legalizedOps1, legalizedOps2, legalizedOps3;

  ConversionTarget target(*context);
  target.addLegalDialect<ONNXDialect, func::FuncDialect, arith::ArithDialect>();

  // Call RewriteONNXForZHigh pass.
  RewritePatternSet Patterns1(context);
  getRewriteONNXForZHighPatterns(Patterns1, &dimAnalysis);
  getRewriteONNXForZHighDynamicallyLegal(&target, &dimAnalysis);
  (void)applyAnalysisConversion(
      module, target, std::move(Patterns1), legalizedOps1);

  // Call ONNXToZHigh pass for lowering multiple ONNX ops at once to ZHigh.
  // E.g. `onnx.ReLu (onnx.Conv)` to zhigh.Conv.
  RewritePatternSet Patterns2(context);
  getONNXToZHighOneOpPatterns(Patterns2);
  (void)applyAnalysisConversion(
      module, target, std::move(Patterns2), legalizedOps2);

  // Call ONNXToZHigh pass for lowering a single ONNX op to ZHigh.
  RewritePatternSet Patterns3(context);
  getONNXToZHighOneOpPatterns(Patterns3);
  getONNXToZHighOneOpDynamicallyLegal(&target, &dimAnalysis);
  (void)applyAnalysisConversion(
      module, target, std::move(Patterns3), legalizedOps3);

  // Get the legalized ops that will run on the host.
  OpSetType cpuOps = llvm::set_intersection<OpSetType, OpSetType>(
      legalizedOps1, llvm::set_intersection<OpSetType, OpSetType>(
                         legalizedOps2, legalizedOps3));

  // Now annotate accelerator operations in the IR with `device` attribute.
  module.walk([&](Operation *op) -> WalkResult {
    if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
      return WalkResult::advance();
    // No annotation for these ops.
    if (isa<ONNXEntryPointOp, ONNXReturnOp, ONNXConstantOp>(op))
      return WalkResult::advance();
    // If `device` is already set, respect it.
    StringAttr device = op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
    if (device && !device.getValue().empty())
      return WalkResult::advance();
    // Op that is legal (should remain on the CPU) as determined by compiler analysis.
    if (cpuOps.contains(op))
      return WalkResult::advance();
    // Now we have an operation that can work on the NNPA, check if its beneficial
    if (useCostBenefitEstimation) {
      // hi alex
    }
    // Compiler determined that we want this op on the NNPA, mark as such.
    op->setAttr(DEVICE_ATTRIBUTE, StringAttr::get(context, NNPA_DEVICE));
    return WalkResult::advance();
  });
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
