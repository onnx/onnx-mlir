// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <set>
#include <string>

using namespace mlir;

namespace {

// Check if a constant tensor is all zeros
bool isAllZeros(DenseElementsAttr attr) {
  if (attr.isSplat()) {
    if (auto floatAttr = mlir::dyn_cast<FloatType>(attr.getElementType())) {
      return attr.getSplatValue<APFloat>().isZero();
    } else if (auto intAttr =
                   mlir::dyn_cast<IntegerType>(attr.getElementType())) {
      return attr.getSplatValue<APInt>().isZero();
    }
  }

  // Check all elements
  if (auto floatAttr = mlir::dyn_cast<FloatType>(attr.getElementType())) {
    return llvm::all_of(attr.getValues<APFloat>(),
                        [](APFloat val) { return val.isZero(); });
  } else if (auto intAttr =
                 mlir::dyn_cast<IntegerType>(attr.getElementType())) {
    return llvm::all_of(attr.getValues<APInt>(),
                        [](APInt val) { return val.isZero(); });
  }

  return false;
}

// Extract DenseElementsAttr from a value (if it's a constant)
std::optional<DenseElementsAttr> getConstantAttr(Value value) {
  if (!value)
    return std::nullopt;

  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (auto constOp = dyn_cast<mlir::ONNXConstantOp>(defOp)) {
    if (auto valueAttr = constOp.getValueAttr()) {
      if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(valueAttr)) {
        return denseAttr;
      }
    }
  }

  return std::nullopt;
}

// Create a zero constant tensor with given shape and type
Value createZeroConstant(PatternRewriter &rewriter, Location loc,
                         RankedTensorType type) {
  DenseElementsAttr zeroAttr;
  if (mlir::isa<FloatType>(type.getElementType())) {
    zeroAttr = DenseElementsAttr::get(
        type, rewriter.getFloatAttr(type.getElementType(), 0.0));
  } else if (mlir::isa<IntegerType>(type.getElementType())) {
    zeroAttr = DenseElementsAttr::get(
        type, rewriter.getIntegerAttr(type.getElementType(), 0));
  } else {
    return nullptr;
  }

  return rewriter.create<mlir::ONNXConstantOp>(
      loc, type, mlir::ValueRange{},
      mlir::ArrayRef<mlir::NamedAttribute>{
          rewriter.getNamedAttr("value", zeroAttr)});
}

/// Remove Mul with zeros
struct RemoveMulWithZerosPattern : public OpRewritePattern<mlir::ONNXMulOp> {
  using OpRewritePattern<mlir::ONNXMulOp>::OpRewritePattern;
  /// match and rewrite Mul with zeros
  LogicalResult matchAndRewrite(mlir::ONNXMulOp mulOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = mulOp.getA();
    Value rhs = mulOp.getB();
    // Check if either input is a zero constant
    auto lhsAttr = getConstantAttr(lhs);
    auto rhsAttr = getConstantAttr(rhs);

    bool hasZero =
        (lhsAttr && isAllZeros(*lhsAttr)) || (rhsAttr && isAllZeros(*rhsAttr));

    if (!hasZero) {
      return failure();
    }

    // Replace with zero constant
    auto resultType = mlir::cast<RankedTensorType>(mulOp.getType());
    Value zero = createZeroConstant(rewriter, mulOp.getLoc(), resultType);

    if (!zero) {
      return failure();
    }

    rewriter.replaceOp(mulOp, zero);

    return success();
  }
};
/// Remove tensor layout unchanged ops (Concat/Reshape)
struct RemoveSemanticallyUselessOps : public RewritePattern {
  /// constructor
  RemoveSemanticallyUselessOps(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  /// match and rewrite tensor layout unchanged ops (Concat/Reshape)
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Check if operation is in the list of tensor-unchanged ops
    std::set<std::string> tensorUnchangeOpList = {"onnx.Concat", "onnx.Reshape",
                                                  "onnx.Identity"};
    std::string opName = op->getName().getStringRef().str();

    if (!tensorUnchangeOpList.contains(opName)) {
      return failure();
    }

    // Check if operation has users (fanout ops)
    if (op->getUsers().empty()) {
      return failure();
    }

    // Check if operation has at least one operand and one result
    if (op->getNumOperands() < 1 || op->getNumResults() < 1) {
      return failure();
    }

    // Get input and output types
    Value input = op->getOperand(0);
    Value output = op->getResult(0);

    auto inputType = mlir::dyn_cast<ShapedType>(input.getType());
    auto outputType = mlir::dyn_cast<ShapedType>(output.getType());

    if (!inputType || !outputType) {
      return failure();
    }

    // Check if both have static shapes
    if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    if (inputShape != outputShape) {
      return failure();
    }

    // Replace all uses of the operation's result with its input
    // This will delete the operation, so it must be done LAST
    rewriter.replaceOp(op, input);

    return success();
  }
};
/// Remove Sub with same inputs (result = 0)
struct RemoveSubWithSameInputsPattern
    : public OpRewritePattern<mlir::ONNXSubOp> {
  using OpRewritePattern<mlir::ONNXSubOp>::OpRewritePattern;
  /// match and rewrite Sub with same inputs (result = 0)
  LogicalResult matchAndRewrite(mlir::ONNXSubOp subOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = subOp.getA();
    Value rhs = subOp.getB();

    // Check if both inputs are the same value
    if (lhs != rhs) {
      return failure();
    }

    // Replace with zero constant
    auto resultType = mlir::cast<RankedTensorType>(subOp.getType());
    Value zero = createZeroConstant(rewriter, subOp.getLoc(), resultType);

    if (!zero) {
      return failure();
    }

    rewriter.replaceOp(subOp, zero);

    return success();
  }
};
/// Remove redundant pad (all zeros)
struct RemoveRedundantPadPattern : public OpRewritePattern<mlir::ONNXPadOp> {
  using OpRewritePattern<mlir::ONNXPadOp>::OpRewritePattern;
  /// matchandrewrite redundant pad (all zeros)
  LogicalResult matchAndRewrite(mlir::ONNXPadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Check if pads attribute exists and is all zeros
    auto padsAttr = getConstantAttr(padOp.getPads());
    if (!padsAttr) {
      return failure();
    }

    // Check if all padding values are zero
    bool allZero = true;
    for (auto val : padsAttr->getValues<APInt>()) {
      if (!val.isZero()) {
        allZero = false;
        break;
      }
    }

    if (!allZero) {
      return failure();
    }

    // Remove the pad operation
    rewriter.replaceOp(padOp, padOp.getData());

    return success();
  }
};

/// Pattern 7: Remove Resize with scale=1 or same input/output size
struct RemoveRedundantResizePattern
    : public OpRewritePattern<mlir::ONNXResizeOp> {
  using OpRewritePattern<mlir::ONNXResizeOp>::OpRewritePattern;
  /// match and rewrite redundant resize (scale=1 or same input/output size)
  LogicalResult matchAndRewrite(mlir::ONNXResizeOp resizeOp,
                                PatternRewriter &rewriter) const override {
    auto inputType =
        mlir::dyn_cast<RankedTensorType>(resizeOp.getX().getType());
    auto outputType = mlir::dyn_cast<RankedTensorType>(resizeOp.getType());

    if (!inputType || !outputType) {
      return failure();
    }

    bool removable = false;

    // Check if scales exist and are all 1.0
    if (resizeOp.getScales()) {
      auto scalesAttr = getConstantAttr(resizeOp.getScales());
      if (scalesAttr) {
        bool allOnes = true;
        if (mlir::isa<FloatType>(scalesAttr->getElementType())) {
          for (auto val : scalesAttr->getValues<APFloat>()) {
            if (!val.isExactlyValue(1.0)) {
              allOnes = false;
              break;
            }
          }
          if (allOnes) {
            removable = true;
          }
        }
      }
    }

    // Check if input and output shapes are the same
    if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
      if (inputType.getShape() == outputType.getShape()) {
        removable = true;
      }
    }

    if (!removable) {
      return failure();
    }

    // Remove the resize operation
    rewriter.replaceOp(resizeOp, resizeOp.getX());

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct RemoveSemanticallyUselessOpsPass
    : public PassWrapper<RemoveSemanticallyUselessOpsPass,
                         OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "remove-semantically-useless-ops";
  }
  StringRef getDescription() const override {
    return "Remove semantically useless operations";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveSemanticallyUselessOps>(context);
    patterns.add<RemoveMulWithZerosPattern>(context);
    patterns.add<RemoveSubWithSameInputsPattern>(context);
    patterns.add<RemoveRedundantPadPattern>(context);
    patterns.add<RemoveRedundantResizePattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createRemoveSemanticallyUselessOpsPass() {
  return std::make_unique<RemoveSemanticallyUselessOpsPass>();
}

} // namespace onnx_mlir
