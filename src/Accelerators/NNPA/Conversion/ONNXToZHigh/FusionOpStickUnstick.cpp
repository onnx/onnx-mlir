/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- FusionOpStickUnstick.cpp - Fuse compute op  -----------------===//
//
// Copyright 2023 The IBM Research Authors.
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

using namespace mlir;
using namespace onnx_mlir;
using namespace zhigh;

using OperationSet = std::set<Operation *>;

// TODO: one use respond to false if same output is used twice by a given op.

// TODO: for elementwise/variadic, test that the others are coming from constant
// (ideal) or have no some other suitable condition.
// TODO: also ensure that the pattern did not already apply (aha, but since we
// remove stick/unstick, that should be trivial).
namespace {

// Check a node with type T is fusible or not.
template <typename T>
bool enqueueFusibleOpImpl(Operation *op) {
  return isa<T>(op);
}

// Variadic template to iterate all the Elementwise Ops.
// 1) Termination (void type): do nothing.
template <typename T = void, class... Ts>
bool enqueueFusibleOp(Operation *op);
// 2) Recursion: test first type, and if not successful, recurse.
template <typename T, class... Ts>
bool enqueueFusibleOp(Operation *op) {
  if (enqueueFusibleOpImpl<T>(op))
    return true;
  return enqueueFusibleOp<Ts...>(op);
}
// 3) By default, return false.
template <>
bool enqueueFusibleOp(Operation *op) {
  return false;
}

// TODO: maybe unify the list from Elementwise.cpp and this one.
bool canOpFuseWithStickUnstick(Operation *op) {
  if (!op)
    return false;
  return enqueueFusibleOp<
      // Unary Op
      mlir::ONNXAbsOp, mlir::ONNXAtanOp, mlir::ONNXBinarizerOp,
      mlir::ONNXCastOp, mlir::ONNXCeilOp, mlir::ONNXCosOp, mlir::ONNXCoshOp,
      mlir::ONNXDequantizeLinearOp, mlir::ONNXCeluOp, mlir::ONNXEluOp,
      mlir::ONNXErfOp, mlir::ONNXAcosOp, mlir::ONNXAcoshOp, mlir::ONNXAsinOp,
      mlir::ONNXAsinhOp, mlir::ONNXAtanhOp, mlir::ONNXExpOp, mlir::ONNXFloorOp,
      mlir::ONNXGeluOp, mlir::ONNXBitwiseNotOp, mlir::ONNXHardSigmoidOp,
      mlir::ONNXHardSwishOp, mlir::ONNXIsInfOp, mlir::ONNXIsNaNOp,
      mlir::ONNXLeakyReluOp, mlir::ONNXLogOp, mlir::ONNXMishOp, mlir::ONNXNegOp,
      mlir::ONNXNotOp, mlir::ONNXReciprocalOp, mlir::ONNXReluOp,
      mlir::ONNXRoundOp, mlir::ONNXSeluOp, mlir::ONNXShrinkOp,
      mlir::ONNXSigmoidOp, mlir::ONNXSignOp, mlir::ONNXSinOp, mlir::ONNXSinhOp,
      mlir::ONNXSoftplusOp, mlir::ONNXSoftsignOp, mlir::ONNXSqrtOp,
      mlir::ONNXTanOp, mlir::ONNXTanhOp, mlir::ONNXThresholdedReluOp,
      // Binary Op
      mlir::ONNXBitShiftOp, mlir::ONNXEqualOp, mlir::ONNXGreaterOp,
      mlir::ONNXGreaterOrEqualOp, mlir::ONNXLessOp, mlir::ONNXLessOrEqualOp,
      mlir::ONNXModOp, mlir::ONNXPowOp,
      // Variadic Op
      mlir::ONNXAddOp, mlir::ONNXAndOp, mlir::ONNXDivOp, mlir::ONNXMaxOp,
      mlir::ONNXMeanOp, mlir::ONNXMinOp, mlir::ONNXMulOp, mlir::ONNXOrOp,
      mlir::ONNXSubOp, mlir::ONNXSumOp, mlir::ONNXXorOp>(op);
}

// Give an op that consumes referenceInputVal, check if all the other input
// values have either the same shape for their last dim, or if there is a
// broadcast from the other input value to the reference input value.
bool sameLastDimOrUniBroadcast(
    DimAnalysis *dimAnalysis, Operation *op, Value referenceInputVal) {
  // Get innermost shape of reference input val.
  ShapedType refType = mlir::dyn_cast<ShapedType>(referenceInputVal.getType());
  if (!refType)
    return false;
  int64_t innermostShapeOfRef =
      refType.getShape()[refType.getShape().size() - 1];
  // Now iterate over each of the inputs to op.
  for (Value v : op->getOperands()) {
    if (v == referenceInputVal)
      continue;
    if (dimAnalysis->sameDim(v, -1, referenceInputVal, -1))
      continue;
    // Check if we have a uni-directional broadcast known at compile time.
    ShapedType vType = mlir::dyn_cast<ShapedType>(v.getType());
    int64_t innermostShapeOfV = vType.getShape()[vType.getShape().size() - 1];
    if (!ShapedType::isDynamic(innermostShapeOfRef) && innermostShapeOfV == 1)
      continue;
    // We have a case that we cannot handle.
    return false;
  }
  return true;
}

// When value is consumed by only one op, returns that op. Otherwise return
// null.
Operation *getSingleUseOperationOf(Value val) {
  Operation *singleOp = nullptr;
  for (OpOperand &use : val.getUses()) {
    Operation *useOp = use.getOwner();
    if (singleOp == nullptr)
      singleOp = useOp;
    else if (singleOp != useOp)
      return nullptr;
  }
  return singleOp;
}

// Return compute operation when fusion of unstick -> compute can be fused,
// Nullptr is returned when not feasible. If unstick -> compute -> stick is
// detected, then stickOp is further defined, otherwise it is  nullptr,
Operation *patternForStickUnstickFusionFromUnstick(ZHighUnstickOp unstickOp,
    DimAnalysis *dimAnalysis, ZHighStickOp &stickOpIfSuccessful) {
  stickOpIfSuccessful = nullptr;
  // Investigate if fusion is possible.
  Value unstickVal = unstickOp.getOut();
  // For merge, unstick value can only be used only once.
  Operation *computeOp = getSingleUseOperationOf(unstickVal);
  if (!computeOp)
    return nullptr;
  // Ensure compute op can be used.
  if (!canOpFuseWithStickUnstick(computeOp))
    return nullptr;
  // Ensure that the output of unstick and op have the same shape.
  // This essentially prevent a broadcasting to occur on the unstickified input.
  // Check also that the compute op other inputs are fine (same or uni
  // broadcast).
  Value computeVal = computeOp->getResult(0);
  if (!dimAnalysis->sameShape(unstickVal, computeVal)) {
    llvm::dbgs() << "Unstick->compute (" << computeOp->getName()
                 << ") fusion FAILURE due to output shape:\n  ";
    unstickVal.dump();
    llvm::dbgs() << "  ";
    computeVal.dump();
    return nullptr;
  }
  if (!sameLastDimOrUniBroadcast(dimAnalysis, computeOp, unstickVal)) {
    llvm::dbgs() << "Unstick->compute (" << computeOp->getName()
                 << ") fusion FAILURE due to input shapes:\n  ";
    unstickVal.dump();
    llvm::dbgs() << "  ";
    computeVal.dump();
    return nullptr;
  }

#if 0 
  // Remove it for now, as we can handle the two part separately.

  // Now we have a unstick->compute that can be fused.
  // But investigate first if we can further unify with the stick.
  Operation *nextOp = getSingleUseOperationOf(computeVal);
  if (nextOp) {
    ZHighStickOp stickOp = mlir::dyn_cast<ZHighStickOp>(nextOp);
    if (stickOp) {
      // Now we have a single Stick op as consumer of computeOp.
      if (zhigh::getZTensorLayout(unstickOp.getIn().getType()) ==
          zhigh::getZTensorLayout(stickOp.getOut().getType())) {
        // Now we have the same layout for unstick input and stich output.
        LLVM_DEBUG({
          llvm::dbgs() << "Unstick->compute->stick (" << computeOp->getName()
                       << ") fusion:\n  ";
          unstickVal.dump();
          llvm::dbgs() << "  ";
          computeVal.dump();
          llvm::dbgs() << "  ";
          nextOp->dump();
        });
        stickOpIfSuccessful = stickOp;
        return computeOp;
      }
    }
  }
  // Failed to find a suitable stick.
#endif

  LLVM_DEBUG({
    llvm::dbgs() << "Unstick->compute (" << computeOp->getName()
                 << ") fusion:\n  ";
    unstickVal.dump();
    llvm::dbgs() << "  ";
    computeVal.dump();
  });
  return computeOp;
}

// Return compute operation when compute -> stick can be fused.
Operation *patternForStickUnstickFusionFromStick(
    ZHighStickOp stickOp, DimAnalysis *dimAnalysis) {

  // Input of stick can only be used once.
  Value stickInputVal = stickOp.getIn();
  if (!stickInputVal.hasOneUse())
    return nullptr;

  // Get use operation and ensure it can be used.
  Operation *computeOp = stickInputVal.getDefiningOp();
  if (!canOpFuseWithStickUnstick(computeOp))
    return nullptr;

  // ZHighUnstickOp unstickOp = computeOp->
  LLVM_DEBUG({
    llvm::dbgs() << "Compute->stick (" << computeOp->getName()
                 << ") fusion:\n  ";
    computeOp->dump();
    llvm::dbgs() << "  ";
    stickOp.dump();
  });
  return computeOp;
}

class PatternsStartingFromUnstick : public OpRewritePattern<ZHighUnstickOp> {
public:
  DimAnalysis *dimAnalysis;
  OperationSet *unstickToRemove;

  PatternsStartingFromUnstick(MLIRContext *context, DimAnalysis *dimAnalysis,
      OperationSet *unstickToRemove)
      : OpRewritePattern<ZHighUnstickOp>(context, 1), dimAnalysis(dimAnalysis),
        unstickToRemove(unstickToRemove) {}

  using OpRewritePattern<ZHighUnstickOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ZHighUnstickOp unstickOp, PatternRewriter &rewriter) const override {

    ZHighStickOp stickOp; // hi alex, remove once we commit to the 2 patterns.
    Operation *computeOp = patternForStickUnstickFusionFromUnstick(
        unstickOp, dimAnalysis, stickOp);
    if (computeOp) {
      int64_t operandNum = computeOp->getNumOperands();
      rewriter.modifyOpInPlace(computeOp, [&]() {
        for (int64_t i = 0; i < operandNum; ++i)
          if (computeOp->getOperand(i) == unstickOp.getOut())
            // Have to replace this operand by input of unstick.
            computeOp->setOperand(i, unstickOp.getIn());
      });
      unstickToRemove->insert(unstickOp.getOperation());
      return success();
    }
    return failure();
  }
};

class PatternsEndingWithStick : public OpRewritePattern<ZHighStickOp> {
public:
  DimAnalysis *dimAnalysis;
  OperationSet *stickToRemove;

  PatternsEndingWithStick(MLIRContext *context, DimAnalysis *dimAnalysis,
      OperationSet *stickToRemove)
      : OpRewritePattern<ZHighStickOp>(context, 1), dimAnalysis(dimAnalysis),
        stickToRemove(stickToRemove) {}

  using OpRewritePattern<ZHighStickOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ZHighStickOp stickOp, PatternRewriter &rewriter) const override {

    Operation *computeOp =
        patternForStickUnstickFusionFromStick(stickOp, dimAnalysis);
    if (computeOp) {
      int64_t resultNum = computeOp->getNumResults();
      assert(resultNum == 1 && "expect only one result for any fused ops");
      assert(computeOp->getResult(0) == stickOp.getIn() &&
             "expected ouput to be stick input");
      //rewriter.modifyOpInPlace(computeOp, [&]() {
        computeOp->getResult(0).replaceAllUsesWith(stickOp.getOut());
      //});
      stickToRemove->insert(stickOp.getOperation());
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
    OperationSet stickUnstickToRemove;

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<PatternsStartingFromUnstick>(
        &getContext(), dimAnalysis, &stickUnstickToRemove);
    //patterns.insert<PatternsEndingWithStick>(
    //    &getContext(), dimAnalysis, &stickUnstickToRemove);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();

#if 0
    for (Operation *op : stickUnstickToRemove) {
      op->erase();
    }
#endif
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
