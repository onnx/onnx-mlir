/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZHighLayoutPropagation.cpp - ZHigh High Level Optimizer ---===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

#define DEBUG_TYPE "zhigh-layout-propagation"

// If set to 1, enable multiple distinct layouts to elementwise compute
// operations; 0 otherwise. We can support the "compiler supported" layouts
// because we only care about SIMD gen of the E1 (innermost) dimension.
#define ELEMENTWISE_WITH_MULTIPLE_LAYOUTS 1

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

namespace onnx_mlir {
namespace zhigh {

namespace {

//===----------------------------------------------------------------------===//
// Helper functions for this pass
//===----------------------------------------------------------------------===//

/// Check if all values are produced by ZHighUnstickOp with the same layout.
std::pair<bool, StringAttr> areProducedByUnstickOpSameLayout(
    PatternRewriter &rewriter, ValueRange values) {
  // Check the first value and get its layout.
  Value first = values[0];
  if (mlir::isa<BlockArgument>(first) ||
      !isa<ZHighUnstickOp>(first.getDefiningOp()))
    return std::make_pair(false, nullptr);
  Value firstStickifiedVal =
      mlir::cast<ZHighUnstickOp>(first.getDefiningOp()).getIn();
  StringAttr firstLayout = convertZTensorDataLayoutToStringAttr(
      rewriter, getZTensorLayout(firstStickifiedVal.getType()));

  // Check all values.
  bool allTheSame = llvm::all_of(values, [&](Value v) {
    using namespace onnx_mlir::zhigh;
    if (mlir::isa<BlockArgument>(v) || !isa<ZHighUnstickOp>(v.getDefiningOp()))
      return false;
    Value stickifiedVal = mlir::cast<ZHighUnstickOp>(v.getDefiningOp()).getIn();
    StringAttr nextLayout = convertZTensorDataLayoutToStringAttr(
        rewriter, getZTensorLayout(stickifiedVal.getType()));
    return (nextLayout == firstLayout);
  });

  if (allTheSame)
    return std::make_pair(true, firstLayout);
  return std::make_pair(false, nullptr);
}

/// Return zTensors that are unstickified into the given tensors.
SmallVector<Value, 4> getZTensors(
    PatternRewriter &rewriter, Location loc, ValueRange tensors) {
  SmallVector<Value, 4> zTensors;
  for (Value v : tensors)
    zTensors.emplace_back(v.getDefiningOp()->getOperands()[0]);
  return zTensors;
}

/// Return a zTensorType for the given tensor and layout.
Type getZTensorType(
    PatternRewriter &rewriter, Location loc, Value tensor, StringAttr layout) {
  // Borrow ZHighStickOp to infer a zTensor type.
  ZHighStickOp stickOp =
      rewriter.create<ZHighStickOp>(loc, tensor, layout, IntegerAttr());
  (void)stickOp.inferShapes([](Region &region) {});

  Type returnType = stickOp.getOut().getType();
  rewriter.eraseOp(stickOp);

  return returnType;
}

//===----------------------------------------------------------------------===//
// ZHigh layout propagation patterns
//===----------------------------------------------------------------------===//

/// The pattern
///   onnx.Concat (zhigh.Unstick (%X1), zhigh.Unstick (%X2)) { axis })
/// can be replaced by
///   zhigh.Unstick (onnx.Concat (%X1, %X2) { new_axis })
class ONNXConcatLayoutPropagatePattern : public OpRewritePattern<ONNXConcatOp> {
public:
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConcatOp concatOp, PatternRewriter &rewriter) const override {
    Operation *genericOp = concatOp.getOperation();
    Location loc = genericOp->getLoc();
    ValueRange inputs = concatOp.getInputs();
    IntegerAttr axis = concatOp.getAxisAttr();
    Value output = concatOp.getConcatResult();

    // Variables for capturing values and attributes used while creating ops
    StringAttr layout;
    bool allTheSame;
    std::tie(allTheSame, layout) =
        areProducedByUnstickOpSameLayout(rewriter, inputs);
    if (!allTheSame)
      return failure();

    // Only support LAYOUT_4D and LAYOUT_NHWC at this moment. They have the same
    // stickification scheme.
    if (!(isNHWCLayout(layout) || is4DLayout(layout)))
      return failure();

    if (!haveNoPadsWhenStickified(inputs, layout, axis))
      return failure();

    // Rewrite
    SmallVector<Value, 4> tblgen_repl_values;
    SmallVector<Value, 4> zTensors = getZTensors(rewriter, loc, inputs);
    IntegerAttr newAxis = getNewConcatAxis(rewriter, layout, axis);
    Type newOutputType = getZTensorType(rewriter, loc, output, layout);

    Value zOutput =
        rewriter.create<ONNXConcatOp>(loc, newOutputType, zTensors, newAxis);
    Value replacedValue =
        rewriter.create<ZHighUnstickOp>(loc, output.getType(), zOutput);
    rewriter.replaceOp(genericOp, replacedValue);
    return ::mlir::success();
  };

private:
  // Check if there are no pads along the given axis when stickifying values by
  // using the given layout.
  bool haveNoPadsWhenStickified(
      ValueRange values, StringAttr layoutAttr, IntegerAttr axisAttr) const {
    if (!layoutAttr)
      return false;
    // Only support LAYOUT_4D and LAYOUT_NHWC at this moment. They have the same
    // stickification scheme.
    if (!(isNHWCLayout(layoutAttr) || is4DLayout(layoutAttr)))
      return false;
    // Only support C dimension at this moment.
    int CAxis = 3; // C is at 3 for 4D and NHWC.
    if (isNHWCLayout(layoutAttr))
      // Value is NCHW that will be directly stickified to NHWC. So C is at 1.
      CAxis = 1;
    if (axisAttr.getValue().getSExtValue() != CAxis)
      return false;

    // C dimension is tiled by 64 when stickified. Hence, checking `C mod 64`
    // for padding.
    // TODO: get this info from affine_map that is used for stickiyfing NHWC.
    return llvm::all_of(values, [&layoutAttr](Value v) {
      if (mlir::isa<ShapedType>(v.getType()) &&
          mlir::cast<ShapedType>(v.getType()).hasRank()) {
        ArrayRef<int64_t> dims = mlir::cast<ShapedType>(v.getType()).getShape();
        if (isNHWCLayout(layoutAttr))
          // Value is NCHW that will be directly unstickified from NHWC.
          // NCHW, C is at 1.
          return (dims[1] % 64 == 0);
        else
          // 4D (similar to NHWC), C is at 3.
          return (dims[3] % 64 == 0);
      }
      return false;
    });
  }

  IntegerAttr getNewConcatAxis(PatternRewriter &rewriter, StringAttr layout,
      IntegerAttr axisAttr) const {
    int axis = axisAttr.getValue().getSExtValue();
    if (isNHWCLayout(layout)) {
      SmallVector<int, 4> NCHWtoNHWC = {0, 3, 1, 2};
      return rewriter.getIntegerAttr(
          rewriter.getIntegerType(64, true), NCHWtoNHWC[axis]);
    }
    return axisAttr;
  }
};

//===----------------------------------------------------------------------===//
// ZHigh layout propagation patterns for elementwise ops
//===----------------------------------------------------------------------===//

using OperationSet = std::set<Operation *>;

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
  OperationSet *processedUnstick;

  PatternsStartingFromUnstick(MLIRContext *context, DimAnalysis *dimAnalysis,
      OperationSet *processedUnstick)
      : OpRewritePattern<ZHighUnstickOp>(context, 1), dimAnalysis(dimAnalysis),
        processedUnstick(processedUnstick) {}

  using OpRewritePattern<ZHighUnstickOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ZHighUnstickOp unstickOp, PatternRewriter &rewriter) const override {

    // Process each stick/unstick only once.
    if (processedUnstick->find(unstickOp.getOperation()) !=
        processedUnstick->end())
      return failure();

    Operation *computeOp = patternForFusionFromUnstick(unstickOp, dimAnalysis);
    if (computeOp) {
      int64_t operandNum = computeOp->getNumOperands();
      rewriter.modifyOpInPlace(computeOp, [&]() {
        for (int64_t i = 0; i < operandNum; ++i)
          if (computeOp->getOperand(i) == unstickOp.getOut())
            // Have to replace this operand by input of unstick.
            computeOp->setOperand(i, unstickOp.getIn());
      });
      processedUnstick->insert(unstickOp.getOperation());
      return success();
    }
    return failure();
  }
};

class PatternsEndingWithStick : public OpRewritePattern<ZHighStickOp> {
public:
  DimAnalysis *dimAnalysis;
  OperationSet *processedStick;

  PatternsEndingWithStick(MLIRContext *context, DimAnalysis *dimAnalysis,
      OperationSet *processedStick)
      : OpRewritePattern<ZHighStickOp>(context, 1), dimAnalysis(dimAnalysis),
        processedStick(processedStick) {}

  using OpRewritePattern<ZHighStickOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ZHighStickOp stickOp, PatternRewriter &rewriter) const override {

    // Process each stick/unstick only once.
    if (processedStick->find(stickOp.getOperation()) != processedStick->end())
      return failure();

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
      // Keep track of the removed ops.
      processedStick->insert(stickOp.getOperation());
      return success();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// ZHigh layout propagation from table gen.
//===----------------------------------------------------------------------===//

/// Use anonymous namespace to avoid duplication symbol `populateWithGenerated`
/// among multiple tablegen-based definitions.

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Transform/ZHigh/ONNXZHighLayoutPropagation.inc"

//===----------------------------------------------------------------------===//
// ZHigh layout propagation Pass
//===----------------------------------------------------------------------===//

struct ZHighLayoutPropagationPass
    : public PassWrapper<ZHighLayoutPropagationPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighLayoutPropagationPass)

  StringRef getArgument() const override { return "zhigh-layout-prop"; }

  StringRef getDescription() const override {
    return "Layout propagation at ZHighIR.";
  }

  ZHighLayoutPropagationPass() = default;
  ZHighLayoutPropagationPass(const ZHighLayoutPropagationPass &pass)
      : PassWrapper<ZHighLayoutPropagationPass, OperationPass<ModuleOp>>() {}
  ZHighLayoutPropagationPass(bool disableElementwiseLayoutProp) {
    this->disableElementwiseLayoutProp = disableElementwiseLayoutProp;
  }

  Option<bool> disableElementwiseLayoutProp{*this,
      "disable-elementwise-layout-propagation",
      llvm::cl::desc("Disable the propagation of ZHigh layouts into "
                     "elementwise operations"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Layout propagation for ZHigh Ops.
    populateWithGenerated(patterns);

    // Concat
    patterns.insert<ONNXConcatLayoutPropagatePattern>(&getContext());

    // Elementwise operations driven from stick/unstick.
    DimAnalysis *dimAnalysis = nullptr;
    OperationSet processedStickUnstick;

    if (!disableElementwiseLayoutProp) {
      dimAnalysis = new DimAnalysis(module);
      dimAnalysis->analyze();
      patterns.insert<PatternsStartingFromUnstick>(
          &getContext(), dimAnalysis, &processedStickUnstick);
      patterns.insert<PatternsEndingWithStick>(
          &getContext(), dimAnalysis, &processedStickUnstick);
    }

    // We want to canonicalize stick/unstick ops during this pass to simplify
    // rules in this pass.
    ZHighStickOp::getCanonicalizationPatterns(patterns, &getContext());
    ZHighUnstickOp::getCanonicalizationPatterns(patterns, &getContext());
    (void)applyPatternsGreedily(module, std::move(patterns));
  }
};
} // anonymous namespace

std::unique_ptr<Pass> createZHighLayoutPropagationPass(
    bool disableElementwiseLayoutProp) {
  return std::make_unique<ZHighLayoutPropagationPass>(
      disableElementwiseLayoutProp);
}

} // namespace zhigh
} // namespace onnx_mlir
