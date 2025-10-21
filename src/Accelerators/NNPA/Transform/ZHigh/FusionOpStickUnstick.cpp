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
#define ENABLE_ELEMENTWISE_WITH_MULTIPLE_LAYOUTS 1

// If set to 1, we will perform fusion even when the unstick is fed to a compute
// operation that will broadcast that values, thus resulting in multiple
// unstickification of a given data element.
#define ENABLE_UNSTICK_BROADCAST_TO_ELEMENTWISE 0

using namespace mlir;
using namespace onnx_mlir;
using namespace zhigh;

namespace {

//===----------------------------------------------------------------------===//
// Utilities to report info.

// Helper function for stick/unstick, do not use directly.
template <typename OP>
static void explanationStickUnstick(Operation *computeOp, OP stickOrUnstickOp,
    std::string message, bool success) {
  std::string text = (success ? "SUCCESS: " : "FAILURE: ") + message;
  if constexpr (std::is_same<OP, ZHighStickOp>::value) {
    llvm::dbgs() << "Compute->stick (" << computeOp->getName() << ") fusion "
                 << message << ":\n  ";
    computeOp->dump();
    llvm::dbgs() << "  ";
    stickOrUnstickOp.dump();
  } else if constexpr (std::is_same<OP, ZHighUnstickOp>::value) {
    llvm::dbgs() << "Unstick->compute (" << computeOp->getName() << ") fusion "
                 << message << ":\n  ";
    stickOrUnstickOp.dump();
    llvm::dbgs() << "  ";
    computeOp->dump();
  } else {
    // Ignore second op.
    llvm::dbgs() << "[Unstick->] compute [->Stick] (" << computeOp->getName()
                 << ") fusion " << message << ":\n  ";
    computeOp->dump();
  }
}

// Reporting for stick/unstick fusion.
template <typename OP>
bool notifyFailure(
    Operation *computeOp, OP stickOrUnstickOp, std::string message) {
  LLVM_DEBUG(
      explanationStickUnstick(computeOp, stickOrUnstickOp, message, false));
  return false;
}

template <typename OP>
bool notifySuccess(
    Operation *computeOp, OP stickOrUnstickOp, std::string message) {
  LLVM_DEBUG(explanexplanationStickUnstickation(
      computeOp, stickOrUnstickOp, message, true));
  return true;
}

// Helper function for layout transform, do not use directly.
static void explanationLayoutTransform(
    ONNXLayoutTransformOp layoutTransformOp, Operation *otherOp,
    std::string message, bool success) {
  std::string text = (success ? "SUCCESS: " : "FAILURE: ") + message;
  if (otherOp) {
    llvm::dbgs() << "LayoutTransform and " << otherOp->getName() << " fusion "
                 << message << ":\n  ";
    layoutTransformOp.dump();
    llvm::dbgs() << "  ";
  } else {
    llvm::dbgs() << "LayoutTransform fusion " << message << ":\n  ";
    layoutTransformOp.dump();
  }
}

// Reporting for layout transform.
bool notifyFailure(ONNXLayoutTransformOp &layoutTransformOp, Operation *otherOp,
    std::string message) {
  LLVM_DEBUG(explanationLayoutTransform(
      layoutTransformOp, otherOp, message, false));
  return false;
}

bool notifySuccess(ONNXLayoutTransformOp &layoutTransformOp, Operation *otherOp,
    std::string message) {
  LLVM_DEBUG(explanationLayoutTransform(
      layoutTransformOp, otherOp, message, true));
  return true;
}

//===----------------------------------------------------------------------===//
// Check if operation is compatible

// Specific code for ops of the family of Layer Normalization.
template <typename LAYER_NORM_OP>
bool isLayerNormCompatible(LAYER_NORM_OP layerNorm) {
  Value X = layerNorm.getX();
  int64_t xRank = getRank(X.getType());
  Operation *op = layerNorm.getOperation();
  int64_t axis = layerNorm.getAxis();
  axis = (axis < 0) ? axis + xRank : axis;
  assert(axis >= 0 && axis < xRank && "out of bound layer norm axis");
  if (axis != xRank - 1)
    return notifyFailure(op, op, "LayerNorm with axis != last dim");
  // At this time, restrict cases with innermost dim that is static and multiple
  // of 64.
  int64_t lastDimShape = getShape(X.getType(), -1);
  if (lastDimShape == ShapedType::kDynamic)
    return notifyFailure(op, op, "LayerNorm last dim not static-shaped");
  if (lastDimShape % 64 != 0)
    // Could handle non-multiple of 64, not needed at this time.
    return notifyFailure(op, op, "LayerNorm last dim not multiple of 64");
  // At this time, restrict cases with only one output (others should be none).
  int64_t resNum = op->getNumResults();
  for (int64_t r = 1; r < resNum; ++r) {
    if (!mlir::isa<NoneType>(op->getResult(r).getType()))
      return notifyFailure(
          op, op, "LayerNorm additional outputs not supported");
  }
  return true;
}

static bool canOpFuseWithStickUnstick(Operation *op) {
  if (!op)
    return false;
  // Operations not handled:
  // o no cast as they may have different input sizes/output sizes.
  if (mlir::isa<ONNXCastOp>(op)) {
    return false;
  }

  // Elementwise operations are supported (they have a single output).
#define ELEMENTWISE_ALL(_OP_TYPE)                                              \
  if (mlir::isa<_OP_TYPE>(op))                                                 \
    return true;
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp"

  // Layer normalization type of operations are conditionally accepted.
  // Additional outputs should be NoneType.
  if (auto layerNormOp = mlir::dyn_cast<ONNXLayerNormalizationOp>(op))
    return isLayerNormCompatible(layerNormOp);
  if (auto RMSLayerNormOp = mlir::dyn_cast<ONNXRMSLayerNormalizationOp>(op))
    return isLayerNormCompatible(RMSLayerNormOp);

  // Not supported, fail.
  return false;
}

#if !ENABLE_ELEMENTWISE_WITH_MULTIPLE_LAYOUTS
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
// only support f32 or dlf16 in stickified format. None type is also tolerated.
static bool suitableComputeType(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isF32())
    return true;
  if (elementType.isF16() && isZTensor(type))
    return true;
  if (mlir::isa<NoneType>(elementType))
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
  if (refType.getRank() <= 1) {
    return true; // Scalar would always have static broadcasts.
  }
  int64_t innermostShapeOfRef = getShape(refType, -1);
  // Now iterate over each of the inputs to op.
  for (Value v : op->getOperands()) {
    // Ignore none types
    if (mlir::isa<NoneType>(v.getType()))
      continue;
    // Same dimension, we are fine.
    if (v == referenceInputVal ||
        dimAnalysis->sameDim(v, -1, referenceInputVal, -1))
      continue;
    // Check if we have a uni-directional broadcast known at compile time.
    ShapedType vType = mlir::dyn_cast<ShapedType>(v.getType());
    if (!vType)
      return false;
    if (vType.getRank() <= 1)
      continue; // scalar, static broadcast known.
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

void getKnownBroadcastSize(DimAnalysis *dimAnalysis, Value val, Value refVal,
    int64_t &staticBroadcastSize, int64_t &dynamicBroadcastDimNum) {
  // Both are assumed to have known shapes.
  ArrayRef<int64_t> shape = getShape(val.getType());
  ArrayRef<int64_t> refShape = getShape(refVal.getType());
  int64_t rank = shape.size();
  int64_t refRank = refShape.size();
  assert(rank <= refRank && "expected rank <= reference rank");
  int64_t offset = refRank - rank;
  // Compute broadcast info.
  staticBroadcastSize = 1;
  dynamicBroadcastDimNum = 0;
  for (int64_t refD = 0; refD < refRank; ++refD) {
    int64_t refDim = refShape[refD];
    int64_t d = refD - offset;
    if (d < 0) {
      // We only have output dimensions... pure broadcast
      if (ShapedType::isDynamic(refDim))
        dynamicBroadcastDimNum++;
      else
        staticBroadcastSize *= refDim;
    } else {
      // We have two dimensions, test if they are different.
      int64_t dim = shape[d];
      if (!dimAnalysis->sameDim(val, d, refVal, refD)) {
        if (!ShapedType::isDynamic(refDim) && !ShapedType::isDynamic(dim)) {
          // Different dims are statics, test if stick broadcast to output dim.
          if (dim == 1 && refDim != 1)
            staticBroadcastSize *= refDim;
        }
      }
    }
  }
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

//===----------------------------------------------------------------------===//
// Check pattern starting at Unstick Op.

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
    notifyFailure(/* usefull to find new opportunities not supported yet*/
        computeOp, unstickOp, "compute op cannot fuse");
    return nullptr;
  }
  if (!suitableComputeType(computeOp)) {
    notifyFailure(computeOp, unstickOp, "due to non f32/dlf16 element type");
    return nullptr;
  }
  // We must support this layout.
  if (isZTensor(unstickInVal.getType()) &&
      !supportedLayoutForCompilerGeneratedStickUnstick(
          unstickInVal, /*support NHWC*/ false)) {
    notifyFailure(computeOp, unstickOp, "due to unstick layout");
    return nullptr;
  }
  // Suitable shapes?
  if (!sameLastDimOrStaticBroadcast(dimAnalysis, computeOp, unstickOutVal)) {
    notifyFailure(computeOp, unstickOp, "due to input shapes");
    return nullptr;
  }
#if !ENABLE_ELEMENTWISE_WITH_MULTIPLE_LAYOUTS
  // Suitable layout?
  ZTensorEncodingAttr::DataLayout unstickLayout =
      onnx_mlir::zhigh::getZTensorLayout(unstickInVal.getType());
  if (!suitableLayout(computeOp, unstickLayout)) {
    notifyFailure(computeOp, unstickOp, "due to input zTensor layouts");
    return nullptr;
  }
#endif
#if !ENABLE_UNSTICK_BROADCAST_TO_ELEMENTWISE
  int64_t staticBroadcastSize, dynamicBroadcastDimNum;
  getKnownBroadcastSize(dimAnalysis, unstickOutVal, computeOp->getResult(0),
      staticBroadcastSize, dynamicBroadcastDimNum);
  if (staticBroadcastSize > 1 || dynamicBroadcastDimNum > 0) {
    notifyFailure(computeOp, unstickOp,
        "due to unstick output being broadcasted to compute op");
    return nullptr;
  }
#endif
  // Success.
  notifySuccess(computeOp, unstickOp, "");
  return computeOp;
}

//===----------------------------------------------------------------------===//
// Check pattern starting at Stick Op

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
    if (!mlir::isa<ONNXConstantOp>(computeOp))
      // No explanation for constants...
      notifyFailure(computeOp, stickOp, "compute op cannot fuse");
    return nullptr;
  }
  if (!suitableComputeType(computeOp)) {
    notifyFailure(computeOp, stickOp, "due to non f32/dlf16 element type");
    return nullptr;
  }
  // We must support this layout.
  if (isZTensor(stickOutVal.getType()) &&
      !supportedLayoutForCompilerGeneratedStickUnstick(
          stickOutVal, /*support NHWC*/ false)) {
    notifyFailure(computeOp, stickOp, "due to stick layout");
    return nullptr;
  }
  // Suitable shapes? Has to do it here too as we need to be able to generate
  // code for the computeOp (including the handling of all its inputs).
  if (!sameLastDimOrStaticBroadcast(dimAnalysis, computeOp, stickInVal)) {
    notifyFailure(computeOp, stickOp, "due to input shapes");
    return nullptr;
  }
#if !ENABLE_ELEMENTWISE_WITH_MULTIPLE_LAYOUTS
  ZTensorEncodingAttr::DataLayout stickLayout =
      onnx_mlir::zhigh::getZTensorLayout(stickOp.getOut().getType());
  if (!suitableLayout(computeOp, stickLayout)) {
    notifyFailure(computeOp, stickOp, "due to input zTensor layouts");
    return nullptr;
  }
#endif
  // ZHighUnstickOp unstickOp = computeOp
  notifySuccess(computeOp, stickOp, "");
  return computeOp;
}

//===----------------------------------------------------------------------===//
// Patterns for stick / unstick.
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
      // Fuse only first result of an op (ok for LayerNorm and elementwise ops).
      if (computeOp->getResult(0) != stickOp.getIn()) {
        notifyFailure(
            computeOp, stickOp, "fuse only first result of compute op");
        return failure();
      }
      // New compute op: has type of the stick output.
      int64_t resultNum = computeOp->getNumResults();
      mlir::SmallVector<Type> newResultTypes;
      newResultTypes.emplace_back(
          llvm::dyn_cast<RankedTensorType>(stickOp.getOut().getType()));
      for (int64_t r = 1; r < resultNum; ++r) {
        // Additional output should only be NoneType at this time.
        newResultTypes.emplace_back(computeOp->getResult(r).getType());
      }
      // Clone compute state, insert in regions, and create.
      OperationState state(computeOp->getLoc(),
          computeOp->getName().getStringRef(), computeOp->getOperands(),
          newResultTypes, computeOp->getAttrs());
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

//===----------------------------------------------------------------------===//
// Patterns Layout Transform.

bool hasStaticInnermostDimWithMod(Value val, int64_t mod) {
  // First constraint for ZHighExtendedLayoutTransformOp
  if (!hasShapeAndRank(val))
    return false;
  ShapedType type = mlir::cast<ShapedType>(val.getType());
  auto shape = type.getShape();
  int64_t rank = type.getRank();
  if (rank == 0)
    return false; // Is a scalar.
  if (shape[rank - 1] == ShapedType::kDynamic)
    return false; // Non-static.
  if (mod > 1 && shape[rank - 1] % mod != 0)
    return false; // Does not satisfy mod constraint.
  return true;
}

bool doesTransposeLeaveInnermostInPlace(mlir::ArrayAttr &permute) {
  int64_t rank = ArrayAttrSize(permute);
  return ArrayAttrIntVal(permute, rank - 1) == rank - 1;
}

class PatternsForExtendedLayoutTransform
    : public OpRewritePattern<ZHighExtendedLayoutTransformOp> {
public:
  DimAnalysis *dimAnalysis;

  PatternsForExtendedLayoutTransform(
      MLIRContext *context, DimAnalysis *dimAnalysis)
      : OpRewritePattern<ZHighExtendedLayoutTransformOp>(context, 1),
        dimAnalysis(dimAnalysis) {}

  using OpRewritePattern<ZHighExtendedLayoutTransformOp>::OpRewritePattern;

  bool locatePattern(ONNXLayoutTransformOp layoutTransformOp) {
    // First layout transform should target CPU.
    auto originalTargetLayout = layoutTransformOp.getTargetLayout();
    if (originalTargetLayout.has_value())
      return notifyFailure(layoutTransformOp, nullptr,
          "First layout should target CPU"); // No layout == CPU layout.
    Value inputData = layoutTransformOp.getData();
    int64_t inputRank = getRank(inputData.getType());
    if (!isZTensor(inputData.getType()))
      return notifyFailure(
          layoutTransformOp, nullptr, "Expected zTensor input");
    if (!supportedLayoutForCompilerGeneratedStickUnstick(
            inputData, /*nhwc*/ false))
      return notifyFailure(
          layoutTransformOp, nullptr, "Compiler unsupported zTensor input");

    // Parameters for op (if successful).
    int64_t reshapeSplitAxis = -1, reshapeSplitFactor = 1,
            reshapeMergeAxis = -1;
    bool hasTranspose = false, dlf16To32 = false, hasFinalLayout = false;
    mlir::ArrayAttr transposePattern;
    mlir::StringAttr finalLayout;

    // look for a permute
    Value currOutputVal = layoutTransform.getOutput();
    Operation *reshapeSplitOp = useOnlyBy<ONNXReshapeOp>(currOutput);
    if (reshapeSplitOp) {
      ONNXReshapeOp reshapeSplit = mlir::cast<ONNXReshapeOp>(reshapeSplitOp);
      // Do we have a split?
      Value reshapedVal = reshapeSplit.getReshaped();
      int64_t reshapedRank = getRank(reshapedVal.getType());
      if (reshapedRank != inputRank + 1)
        return notifyFailure(layoutTransformOp, reshapeSplitOp,
            "Reshape expected to split one dim (ranks)");
      // Look for the different shapes.
      int64_t dout = 0;
      for (int64_t din = 0; din < inputRank; ++din) {
        if (do >= reshapedRank)
          return notifyFailure(layoutTransformOp, reshapeSplitOp,
              "Reshape expected to split one dim (out of dout)");
        if (dimAnalysis(currOutputVal, din, reshapedVal, dout)) {
          ++dout continue;
        }
        // Since we assume a split of one shape into two: speculatively set the
        // split here.
        if (splitAxis != -1)
          return notifyFailure(layoutTransformOp, reshapeSplitOp,
              "Reshape expected to split one dim (second split)");

        splitAxis = din;
        dout += 2;
      }
    }
  }

  LogicalResult matchAndRewrite(
      ZHighExtendedLayoutTransformOp layoutTransformOp,
      PatternRewriter &rewriter) const override {
    return success();
  }
};
//===----------------------------------------------------------------------===//
// Pass.

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
