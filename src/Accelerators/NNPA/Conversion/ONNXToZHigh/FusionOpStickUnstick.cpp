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
// Device placement is done via setting `device` attribute for each operation.
// Values for `device` attribute is one of the following strings:
// - "": an empty string means the compiler decides whether the operation is on
// CPU or NNPA.
// - "nnpa": the operation may run on NNPA or CPU, and the final decision is
// made by the compiler. If `device=nnpa` is the result of this device-placement
// pass, then it means the compiler thinks it is suitable for NNPA.
// - "cpu": the operation is guaranteed to run on CPU.
//
//===----------------------------------------------------------------------===//

#include <regex>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "fusion-op-stick-unstick"

using namespace mlir;
using namespace onnx_mlir;
using namespace zhigh;

// TODO: one use respond to false if same output is used twice by a given op.

namespace {

bool canOpFuseWithStickUnstick(Operation *op) {
  if (!op)
    return false;
  if (llvm::isa<ONNXAddOp>(op))
    return true;
  return false;
}

class PatternsStartingFromUnstick : public OpRewritePattern<ZHighUnstickOp> {
public:
  DimAnalysis *dimAnalysis;

  PatternsStartingFromUnstick(MLIRContext *context, DimAnalysis *dimAnalysis)
      : OpRewritePattern<ZHighUnstickOp>(context, 1), dimAnalysis(dimAnalysis) {
  }

  using OpRewritePattern<ZHighUnstickOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ZHighUnstickOp unstickOp, PatternRewriter &rewriter) const override {

    Value unstickVal = unstickOp.getOut();
    // For merge, unstick can only be used once.
    if (!unstickVal.hasOneUse())
      return failure();
    // Get use operation and ensure it can be used.
    Operation *computeOp = *unstickOp->getUsers().begin();
    if (!canOpFuseWithStickUnstick(computeOp))
      return failure();
    // Ensure that the output of unstick and op have the same shape.
    Value computeVal = computeOp->getResult(0);
    if (!dimAnalysis->sameShape(unstickVal, computeVal))
      return failure();
    // Now we have a unstick->compute that can be fused.
    if (computeVal.hasOneUse()) {
      Operation *nextOp = *computeOp->getUsers().begin();
      if (mlir::isa<ZHighStickOp>(nextOp)) {
        // Has a single Stick op as consumer of computeOp.
        LLVM_DEBUG({
          llvm::dbgs() << "Unstick->compute->stick fusion:\n  ";
          unstickVal.dump();
          llvm::dbgs() << "  ";
          computeVal.dump();
          llvm::dbgs() << "  ";
          nextOp->dump();
        });
        return failure(); // hi alex, success once we have a transformation.
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "Unstick->compute fusion:\n  ";
      unstickVal.dump();
      llvm::dbgs() << "  ";
      computeVal.dump();
    });
    return failure(); // hi alex, success once we have a transformation.
  }
};

class PatternsEndingWithStick : public OpRewritePattern<ZHighStickOp> {
public:
  DimAnalysis *dimAnalysis;

  PatternsEndingWithStick(MLIRContext *context, DimAnalysis *dimAnalysis)
      : OpRewritePattern<ZHighStickOp>(context, 1), dimAnalysis(dimAnalysis) {}

  using OpRewritePattern<ZHighStickOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ZHighStickOp stickOp, PatternRewriter &rewriter) const override {

    // Input of stick can only be used once.
    Value stickInputVal = stickOp.getIn();
    if (!stickInputVal.hasOneUse())
      return failure();

    // Get use operation and ensure it can be used.
    Operation *computeOp = stickInputVal.getDefiningOp();
    if (!canOpFuseWithStickUnstick(computeOp))
      return failure();

    LLVM_DEBUG({
      llvm::dbgs() << "compute->stick fusion:\n  ";
      computeOp->dump();
      llvm::dbgs() << "  ";
      stickOp.dump();
    });
    return failure(); // hi alex, success once we have a transformation.
  }
};

struct FusionOpStickUnstick
    : public PassWrapper<FusionOpStickUnstick, OperationPass<ModuleOp>> {
  using OpSetType = DenseSet<Operation *>;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionOpStickUnstick)

  FusionOpStickUnstick() = default;
  FusionOpStickUnstick(const FusionOpStickUnstick &pass)
      : PassWrapper<FusionOpStickUnstick, OperationPass<ModuleOp>>() {}

  StringRef getArgument() const override { return "fusion-op-stick-unstick"; }

  StringRef getDescription() const override {
    return "Initiate the fusion of operations (elementwise) with stick/unstick "
           "ops";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    DimAnalysis *dimAnalysis = new DimAnalysis(module);
    dimAnalysis->analyze();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<PatternsStartingFromUnstick>(&getContext(), dimAnalysis);
    patterns.insert<PatternsEndingWithStick>(&getContext(), dimAnalysis);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace onnx_mlir {
namespace zhigh {

/*!
 * Create a DevicePlacement pass.
 */
std::unique_ptr<mlir::Pass> createFusionOpStickUnstick() {
  return std::make_unique<FusionOpStickUnstick>();
}

} // namespace zhigh
} // namespace onnx_mlir
