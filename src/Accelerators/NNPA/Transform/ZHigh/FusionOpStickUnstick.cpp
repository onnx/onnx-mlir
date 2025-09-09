/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- FusionOpStickUnstick.cpp - Fuse compute op  -----------------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This pass detects patterns of unstick -> op -> stick (or subset) and when
// successful, remove the stick/unstick to directly feed the ZTensor to the
// compute operation. Lowering will deal with generating the proper code.
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
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "fusion-op-stick-unstick"

// If set to 1, enable multiple distinct layouts to elementwise compute
// operations; 0 otherwise. We can support the "compiler supported" layouts
// because we only care about SIMD gen of the E1 (innermost) dimension.
#define ELEMENTWISE_WITH_MULTIPLE_LAYOUTS 1

using namespace mlir;
using namespace onnx_mlir;
using namespace zhigh;

// TODO: ensure that if there is already a stickified format in the compute,
// then the new fusion is compatible, aka use the same format.
namespace {

// TODO: maybe unify the list from Elementwise.cpp and this one.
static bool canOpFuseWithStickUnstick(Operation *op) {
  if (!op)
    return false;
  // Exceptions:
  // o no cast as they may have different input sizes/output sizes.
  if (isa<ONNXCastOp>(op))
    return false;
    // Return true for all of the elementwise operations.
#define ELEMENTWISE_ALL(_OP_TYPE)                                              \
  if (isa<_OP_TYPE>(op))                                                       \
    return true;
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp"
  return false;
}

#if !ELEMENTWISE_WITH_MULTIPLE_LAYOUTS
// Make sure that all inputs have either an undefined layout or the same as
// reference layout.
static bool suitableLayout(
    Operation *op, ZTensorEncodingAttr::DataLayout refLayout) {
  // Now iterate over each of the inputs to op.
  for (Value v : op->getOperands()) {
    // Check if we have a layout and if it is compatible.
    ZTensorEncodingAttr::DataLayout vLayout =
        onnx_mlir::zhigh::getZTensorLayout(v.getType());
    if (vLayout == ZTensorEncodingAttr::DataLayout::UNDEFINED ||
        vLayout == refLayout)
      continue;
    // We have a Z layout and its not the same, abort
    return false;
  }
  return true;
}
#endif

// Make sure that all inputs and outputs have the right element type. Currently
// only support f32 or d.
static bool suitableComputeType(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isF32())
    return true;
  if (elementType.isF16() && isZTensor(type))
    return true;
  return false;
}

static bool suitableComputeType(Operation *op) {
  for (Value v : op->getOperands()) {
    if (!suitableComputeType(v.getType()))
      return false;
  }
  for (Value v : op->getResults()) {
    if (!suitableComputeType(v.getType()))
      return false;
  }
  return true;
}

// Give an op that consumes referenceInputVal, check if all the other input
// values have either the same shape for their last dim, or if there is a
// broadcast known at compile time.
bool sameLastDimOrStaticBroadcast(
    DimAnalysis *dimAnalysis, Operation *op, Value referenceInputVal) {
  // Get innermost shape of reference input val.
  ShapedType refType = mlir::dyn_cast<ShapedType>(referenceInputVal.getType());
  if (!refType)
    return false; // Expected shaped type, abort.
  if (refType.getRank() == 1)
    return true; // Unary ops never have broadcasts.
  int64_t innermostShapeOfRef = getShape(refType, -1);
  // Now iterate over each of the inputs to op.
  for (Value v : op->getOperands()) {
    // Same dimension, we are fine.
    if (v == referenceInputVal ||
        dimAnalysis->sameDim(v, -1, referenceInputVal, -1))
      continue;
    // Check if we have a uni-directional broadcast known at compile time.
    ShapedType vType = mlir::dyn_cast<ShapedType>(v.getType());
    if (!vType)
      return false;
    int64_t innermostShapeOfV = getShape(vType, -1);
    if (!ShapedType::isDynamic(innermostShapeOfRef) && innermostShapeOfV == 1)
      continue;
    if (!ShapedType::isDynamic(innermostShapeOfV) && innermostShapeOfRef == 1)
      continue;
    // Not the same size and no uni-broadcasting at compile time, fail.
    return false;
  }
  return true;
}

// When value is consumed by only one op, returns that op. Otherwise return
// null.
Operation *getSingleUseOperationOf(Value val) {
  Operation *singleOp = nullptr;
  int multipleComputeOpNum = 0;
  for (OpOperand &use : val.getUses()) {
    Operation *useOp = use.getOwner();
    if (singleOp == nullptr)
      singleOp = useOp;
    else if (singleOp != useOp) {
      multipleComputeOpNum++;
      LLVM_DEBUG({
        if (multipleComputeOpNum == 1) {
          // TODO, could look into tolerating multiple "supported" ops + some
          // free ops such as "DimOp". Printout here highlights the missing
          // opportunities.
          llvm::dbgs() << "Unstick -> multiple compute ops fusion FAILURE:\n  ";
          val.dump();
          llvm::dbgs() << "  ";
          singleOp->dump();
        }
        llvm::dbgs() << "  ";
        useOp->dump();
      });
    }
  }
  return multipleComputeOpNum ? nullptr : singleOp;
}

static void explanation(
    Operation *computeOp, ZHighUnstickOp unstickOp, std::string message) {
  llvm::dbgs() << "Unstick->compute (" << computeOp->getName() << ") fusion "
               << message << ":\n  ";
  unstickOp.dump();
  llvm::dbgs() << "  ";
  computeOp->dump();
}

static void explanation(
    Operation *computeOp, ZHighStickOp stickOp, std::string message) {
  llvm::dbgs() << "Compute->stick (" << computeOp->getName() << ") fusion "
               << message << ":\n  ";
  computeOp->dump();
  llvm::dbgs() << "  ";
  stickOp.dump();
}

// Return compute operation when fusion of unstick -> compute can be fused,
// Nullptr is returned when not feasible. If unstick -> compute -> stick is
// detected, then stickOp is further defined, otherwise it is  nullptr,
Operation *patternForFusionFromUnstick(
    ZHighUnstickOp unstickOp, DimAnalysis *dimAnalysis) {
  Value unstickInVal = unstickOp.getIn();
  Value unstickOutVal = unstickOp.getOut();
  // For merge, unstick value can only be used only once.
  Operation *computeOp = getSingleUseOperationOf(unstickOutVal);
  // Supported compute op?
  if (!computeOp)
    return nullptr;
  if (!canOpFuseWithStickUnstick(computeOp)) {
    LLVM_DEBUG(/* usefull to find new opportunities not supported yet*/
        explanation(computeOp, unstickOp, "FAILURE compute op cannot fuse"));
    return nullptr;
  }
  if (!suitableComputeType(computeOp)) {
    LLVM_DEBUG(explanation(
        computeOp, unstickOp, "FAILURE due to non f32/dlf16 element type"));
    return nullptr;
  }
  // We must support this layout.
  if (isZTensor(unstickInVal.getType()) &&
      !supportedLayoutForCompilerGeneratedStickUnstick(unstickInVal)) {
    LLVM_DEBUG(
        explanation(computeOp, unstickOp, "FAILURE due to unstick shape"));
    return nullptr;
  }
  // Suitable shapes?
  if (!sameLastDimOrStaticBroadcast(dimAnalysis, computeOp, unstickOutVal)) {
    LLVM_DEBUG(
        explanation(computeOp, unstickOp, "FAILURE due to input shapes"));
    return nullptr;
  }
#if !ELEMENTWISE_WITH_MULTIPLE_LAYOUTS
  // Suitable layout?
  ZTensorEncodingAttr::DataLayout unstickLayout =
      onnx_mlir::zhigh::getZTensorLayout(unstickInVal.getType());
  if (!suitableLayout(computeOp, unstickLayout)) {
    LLVM_DEBUG(explanation(
        computeOp, unstickOp, "FAILURE due to input zTensor layouts"));
    return nullptr;
  }
#endif
  // Success.
  LLVM_DEBUG(explanation(computeOp, unstickOp, ""));
  return computeOp;
}

// Return compute operation when compute -> stick can be fused.
Operation *patternForFusionFromStick(
    ZHighStickOp stickOp, DimAnalysis *dimAnalysis) {
  Value stickInVal = stickOp.getIn();
  Value stickOutVal = stickOp.getOut();
  // Input of stick can only be used once.
  if (!stickInVal.hasOneUse())
    return nullptr;
  // Get use operation and ensure it can be used.
  Operation *computeOp = stickInVal.getDefiningOp();
  if (!computeOp)
    return nullptr;
  if (!canOpFuseWithStickUnstick(computeOp)) {
    LLVM_DEBUG(/* usefull to find new opportunities not supported yet*/
        explanation(computeOp, stickOp, "FAILURE compute op cannot fuse"));
    return nullptr;
  }
  if (!suitableComputeType(computeOp)) {
    LLVM_DEBUG(explanation(
        computeOp, stickOp, "FAILURE due to non f32/dlf16 element type"));
    return nullptr;
  }
  // We must support this layout.
  if (isZTensor(stickOutVal.getType()) &&
      !supportedLayoutForCompilerGeneratedStickUnstick(stickOutVal)) {
    LLVM_DEBUG(explanation(computeOp, stickOp, "FAILURE due to stick layout"));
    return nullptr;
  }
#if !ELEMENTWISE_WITH_MULTIPLE_LAYOUTS
  ZTensorEncodingAttr::DataLayout stickLayout =
      onnx_mlir::zhigh::getZTensorLayout(stickOp.getOut().getType());
  if (!suitableLayout(computeOp, stickLayout)) {
    LLVM_DEBUG(explanation(
        computeOp, stickOp, "FAILURE due to input zTensor layouts"));
    return nullptr;
  }
#endif
  // ZHighUnstickOp unstickOp = computeOp
  LLVM_DEBUG(explanation(computeOp, stickOp, ""));
  return computeOp;
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

    Operation *computeOp = patternForFusionFromUnstick(unstickOp, dimAnalysis);
    if (computeOp) {
      int64_t operandNum = computeOp->getNumOperands();
      rewriter.modifyOpInPlace(computeOp, [&]() {
        for (int64_t i = 0; i < operandNum; ++i)
          if (computeOp->getOperand(i) == unstickOp.getOut())
            // Have to replace this operand by input of unstick.
            computeOp->setOperand(i, unstickOp.getIn());
      });
      return success();
    }
    return failure();
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

    Operation *computeOp = patternForFusionFromStick(stickOp, dimAnalysis);
    if (computeOp) {
      int64_t resultNum = computeOp->getNumResults();
      assert(resultNum == 1 && "expect only one result for any fused ops");
      assert(computeOp->getResult(0) == stickOp.getIn() &&
             "expected ouput to be stick input");
      // New compute op: has type of the stick output.
      auto newResultType =
          llvm::dyn_cast<RankedTensorType>(stickOp.getOut().getType());
      // Clone compute state, insert in regions, and create.
      OperationState state(computeOp->getLoc(),
          computeOp->getName().getStringRef(), computeOp->getOperands(),
          {newResultType}, computeOp->getAttrs());
      for (unsigned i = 0, e = computeOp->getNumRegions(); i < e; ++i)
        state.addRegion();
      Operation *newComputeOp = rewriter.create(state);
      // Where ever stick outputs were used, now it should be the new
      // compute's result.
      rewriter.replaceOp(stickOp.getOperation(), newComputeOp->getResult(0));
      return success();
    }
    return failure();
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
    return "Initiate the fusion of operations (elementwise) with "
           "stick/unstick "
           "ops";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    LLVM_DEBUG({
      llvm::dbgs() << "IR before invoking Fusion of op, stick, and unstick:\n";
      module.print(llvm::dbgs());
    });

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
