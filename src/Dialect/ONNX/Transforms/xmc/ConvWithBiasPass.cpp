// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// ConvWithBiasPass
//
// This pass fuses Add(Conv(A, X, none), constant) -> Conv(A, X, bias)
// When a convolution has no bias and is followed by an Add with a constant,
// we can fold the constant into the convolution as bias.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if the value is from an ONNXNoneOp (represents no bias)
bool isNoValue(Value val) { return val.getDefiningOp<ONNXNoneOp>() != nullptr; }

/// Create a 1D bias tensor from a constant value.
/// If the constant is already 1D, return it as-is.
/// If multi-dimensional, flatten it to 1D.
/// Handles both float and quantized types.
Value create1DBiasFromConstant(
    PatternRewriter &rewriter, Value biasVal, Location loc) {
  auto constOp =
      mlir::dyn_cast_or_null<ONNXConstantOp>(biasVal.getDefiningOp());
  if (!constOp)
    return biasVal;

  auto attr = mlir::dyn_cast_or_null<DenseElementsAttr>(constOp.getValueAttr());
  if (!attr)
    return biasVal;

  auto oldType = mlir::dyn_cast<RankedTensorType>(attr.getType());
  if (!oldType || oldType.getRank() == 1)
    return biasVal; // Already 1D or not ranked

  // Get the result type from the constant op (may have quant type)
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(constOp.getResult().getType());
  if (!resultType)
    return biasVal;

  // Flatten to 1D
  int64_t numElements = attr.getNumElements();
  SmallVector<int64_t, 1> newShape = {numElements};

  // Create new storage type (1D version of the attribute type)
  auto storageElemType = oldType.getElementType();
  auto newStorageType = RankedTensorType::get(newShape, storageElemType);

  // Create new result type (1D version, preserving quant type if present)
  auto resultElemType = resultType.getElementType();
  auto newResultType = RankedTensorType::get(newShape, resultElemType);

  // Reshape the attribute data to 1D (works for any element type)
  auto newAttr = attr.reshape(newStorageType);

  // Create new constant with the reshaped attribute
  auto valueAttr = rewriter.getNamedAttr("value", newAttr);
  auto newConstOp = rewriter.create<ONNXConstantOp>(loc, newResultType,
      mlir::ValueRange{}, mlir::ArrayRef<mlir::NamedAttribute>{valueAttr});

  return newConstOp.getResult();
}

/// Check if bias is compatible with weight (bias size == output channels)
bool isBiasCompatibleWithWeight(Value biasVal, Value weightVal) {
  auto wType = mlir::dyn_cast<RankedTensorType>(weightVal.getType());
  if (!wType || wType.getRank() < 1)
    return false;

  int64_t outChannels = wType.getShape()[0];

  // Bias must be a constant tensor
  auto constOp =
      mlir::dyn_cast_or_null<ONNXConstantOp>(biasVal.getDefiningOp());
  if (!constOp)
    return false;

  auto attr = mlir::dyn_cast_or_null<DenseElementsAttr>(constOp.getValueAttr());
  if (!attr)
    return false;

  // Check total number of elements matches output channels
  int64_t biasSize = attr.getNumElements();
  return biasSize == outChannels;
}

//===----------------------------------------------------------------------===//
// Pattern: ConvWithBias
//===----------------------------------------------------------------------===//

/// Pattern to fuse Add(Conv(X, W, none), Constant) -> Conv(X, W, bias)
struct ConvWithBiasPattern : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    // Get add operands
    Value lhs = addOp.getA();
    Value rhs = addOp.getB();

    // Find which operand is the conv and which is the constant
    Value biasConstant = nullptr;
    ONNXConvOp convOp = nullptr;

    // Try lhs as conv, rhs as constant
    if (auto conv = lhs.getDefiningOp<ONNXConvOp>()) {
      if (rhs.getDefiningOp<ONNXConstantOp>()) {
        convOp = conv;
        biasConstant = rhs;
      }
    }

    // Try rhs as conv, lhs as constant
    if (!convOp) {
      if (auto conv = rhs.getDefiningOp<ONNXConvOp>()) {
        if (lhs.getDefiningOp<ONNXConstantOp>()) {
          convOp = conv;
          biasConstant = lhs;
        }
      }
    }

    // No valid pattern found
    if (!convOp || !biasConstant)
      return failure();

    // Conv must have no bias (NoValue)
    Value convBias = convOp.getB();
    if (!isNoValue(convBias))
      return failure();

    // Check if bias is compatible with weights
    Value weights = convOp.getW();
    if (!isBiasCompatibleWithWeight(biasConstant, weights))
      return failure();

    // ============== Pattern matched! Now perform transformation ==============
    Location loc = addOp.getLoc();

    // Create 1D bias from the constant
    Value bias1D = create1DBiasFromConstant(rewriter, biasConstant, loc);

    // Get original conv attributes
    auto autoPadAttr = convOp.getAutoPadAttr();
    auto dilationsAttr = convOp.getDilationsAttr();
    auto groupAttr = convOp.getGroupAttr();
    auto kernelShapeAttr = convOp.getKernelShapeAttr();
    auto padsAttr = convOp.getPadsAttr();
    auto stridesAttr = convOp.getStridesAttr();

    // Create new conv with bias
    auto newConv = rewriter.create<ONNXConvOp>(loc,
        addOp.getResult().getType(), // Output type from the add
        convOp.getX(),               // Input
        convOp.getW(),               // Weights
        bias1D,                      // New bias
        autoPadAttr, dilationsAttr, groupAttr, kernelShapeAttr, padsAttr,
        stridesAttr);

    // Replace add with new conv
    rewriter.replaceOp(addOp, newConv.getResult());

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct ConvWithBiasPass
    : public PassWrapper<ConvWithBiasPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "conv-with-bias"; }
  StringRef getDescription() const override {
    return "Fuse Add(Conv(A, X, none), constant) -> Conv(A, X, bias)";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvWithBiasPattern>(context);

    GreedyRewriteConfig config;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvWithBiasPass() {
  return std::make_unique<ConvWithBiasPass>();
}

} // namespace onnx_mlir
