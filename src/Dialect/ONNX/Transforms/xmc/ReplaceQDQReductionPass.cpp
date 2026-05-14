// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"

#include "llvm/ADT/SmallVector.h"

#include <numeric>
#include <optional>

using namespace mlir;

// ReplaceQDQReductionPass -- ONNX-MLIR port of xcompiler.git's `shape_to_4d`.
// Canonicalises Q/DQ-bracketed Reduce(Sum/Mean/Max/Min) to rank-4 +
// keep_dims=true (matches the AIE reduction kernel and xmodel emission).
// Cast-template chains are skipped (handled by xcompiler.git's
// `replace_reduce`).

namespace {

llvm::SmallVector<int64_t> getShape(mlir::Value value) {
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank())
    return {};
  return llvm::SmallVector<int64_t>(
      shapedType.getShape().begin(), shapedType.getShape().end());
}

int64_t normalizeAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}

/// Bump every axis in `axes` by `shift` (after normalising to `rank`).
void shiftAxes(
    llvm::SmallVectorImpl<int64_t> &axes, int64_t rank, int64_t shift) {
  for (auto &a : axes)
    a = normalizeAxis(a, rank) + shift;
}

bool isReshapeShapingCompatibleElementType(mlir::Type elementType) {
  return mlir::isa<mlir::FloatType>(elementType) ||
         mlir::isa<mlir::quant::QuantizedType>(elementType) ||
         mlir::isa<mlir::IntegerType>(elementType);
}

/// Extract a constant axes vector from a Reduce op's `axes` operand.
template <typename ONNX_OP>
std::optional<llvm::SmallVector<int64_t>> getConstantAxes(ONNX_OP op) {
  mlir::Value axesOperand = op.getAxes();
  if (!axesOperand)
    return std::nullopt;
  auto constOp = axesOperand.template getDefiningOp<mlir::ONNXConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto denseAttr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(constOp.getValueAttr());
  if (!denseAttr)
    return std::nullopt;
  llvm::SmallVector<int64_t> axes;
  for (auto v : denseAttr.getValues<mlir::APInt>())
    axes.push_back(v.getSExtValue());
  return axes;
}

/// Build a 1-D int64 constant tensor (axes / shape input).
mlir::Value createInt64Const1D(mlir::PatternRewriter &rewriter,
    mlir::Location loc, llvm::ArrayRef<int64_t> values) {
  auto tensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, rewriter.getI64Type());
  auto attr = mlir::DenseIntElementsAttr::get(tensorType, values);
  return rewriter.create<mlir::ONNXConstantOp>(loc, tensorType,
      mlir::ValueRange{},
      mlir::ArrayRef<mlir::NamedAttribute>{
          rewriter.getNamedAttr("value", attr)});
}

/// Build an onnx.Reshape with a constant shape input.
/// Element type (incl. quantized) is carried through.
mlir::Value createReshape(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, llvm::ArrayRef<int64_t> targetShape) {
  auto inputType = mlir::cast<mlir::ShapedType>(input.getType());
  auto outputType =
      mlir::RankedTensorType::get(targetShape, inputType.getElementType());
  mlir::Value shapeConst = createInt64Const1D(rewriter, loc, targetShape);
  return rewriter.create<mlir::ONNXReshapeOp>(
      loc, outputType, input, shapeConst, /*allowzero=*/0);
}

/// True iff the reduction is in a `Dequant -> Reduce -> Quant` chain
/// (single fanout).  Accepts explicit Q/DQ ops or implicit quant tensor types.
template <typename ONNX_OP>
bool isInDequantReduceQuantChain(ONNX_OP op) {
  if (!op.getReduced().hasOneUse())
    return false;
  if (op.getData().template getDefiningOp<mlir::ONNXDequantizeLinearOp>()) {
    mlir::Operation *consumer = *op.getReduced().getUsers().begin();
    if (mlir::isa<mlir::ONNXQuantizeLinearOp>(consumer))
      return true;
  }
  auto inShaped = mlir::dyn_cast<mlir::ShapedType>(op.getData().getType());
  auto outShaped = mlir::dyn_cast<mlir::ShapedType>(op.getReduced().getType());
  return inShaped && outShaped &&
         mlir::isa<mlir::quant::QuantizedType>(inShaped.getElementType()) &&
         mlir::isa<mlir::quant::QuantizedType>(outShaped.getElementType());
}

// ReshapeReduceTo4DPattern: reshape strategy by rank.
//   rank < 4   : prepend size-1 dims, bump axes
//   rank == 4  : pass through (re-emit with keep_dims=true)
//   rank > 4   : drop leading-1 OR collapse middle dims
// A trailing reshape restores the original output shape if needed.
template <typename ONNX_OP>
struct ReshapeReduceTo4DPattern : public mlir::OpRewritePattern<ONNX_OP> {
  using mlir::OpRewritePattern<ONNX_OP>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ONNX_OP op, mlir::PatternRewriter &rewriter) const override {

    // Only fire on the dequant -> reduce -> quant template.
    if (!isInDequantReduceQuantChain(op))
      return mlir::failure();

    mlir::Value input = op.getData();
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape())
      return mlir::failure();
    if (!isReshapeShapingCompatibleElementType(inputType.getElementType()))
      return mlir::failure();

    auto axesOpt = getConstantAxes(op);
    if (!axesOpt || axesOpt->empty())
      return mlir::failure();
    llvm::SmallVector<int64_t> axes = *axesOpt;

    auto inputShape = getShape(input);
    int64_t rank = inputShape.size();

    // Multi-axis is only canonicalisable when axis[0] is the last dim.
    if (axes.size() > 1 && axes[0] != rank - 1)
      return mlir::failure();

    int64_t lastAxis = normalizeAxis(axes.back(), rank);

    // Already canonical (also the idempotency guard for re-visits).
    if (op.getKeepdims() != 0 && rank == 4)
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    auto outElemType = mlir::cast<mlir::ShapedType>(op.getReduced().getType())
                           .getElementType();

    // Pick the reshape strategy and build the reduce input.
    mlir::Value newReduceInput;
    llvm::SmallVector<int64_t> newAxes(axes);
    if (rank < 4) {
      int64_t shift = 4 - rank;
      shiftAxes(newAxes, rank, shift);
      llvm::SmallVector<int64_t> paddedShape(shift, 1);
      paddedShape.insert(
          paddedShape.end(), inputShape.begin(), inputShape.end());
      newReduceInput = createReshape(rewriter, loc, input, paddedShape);
    } else if (rank == 4) {
      newReduceInput = input;
    } else if ((lastAxis == 2 || lastAxis == 3) && inputShape[0] == 1) {
      llvm::SmallVector<int64_t> shrunkShape(
          inputShape.begin() + 1, inputShape.end());
      newReduceInput = createReshape(rewriter, loc, input, shrunkShape);
      shiftAxes(newAxes, rank, -1);
    } else {
      // Collapse middle dims; reduction must be on the last axis.
      if (axes.size() != 1 || lastAxis != rank - 1)
        return mlir::failure();
      llvm::SmallVector<int64_t> collapsedShape{inputShape[0], inputShape[1],
          std::accumulate(inputShape.begin() + 2, inputShape.begin() + rank - 1,
              int64_t{1}, std::multiplies<int64_t>()),
          inputShape[rank - 1]};
      newReduceInput = createReshape(rewriter, loc, input, collapsedShape);
      newAxes.assign({3});
    }

    // Re-emit reduce with keep_dims=true.
    auto newInputShape =
        mlir::cast<mlir::RankedTensorType>(newReduceInput.getType()).getShape();
    llvm::SmallVector<int64_t> newReduceOutShape(
        newInputShape.begin(), newInputShape.end());
    for (int64_t a : newAxes)
      newReduceOutShape[a] = 1;
    auto newReduceOutType =
        mlir::RankedTensorType::get(newReduceOutShape, outElemType);

    mlir::Value newAxesVal = createInt64Const1D(rewriter, loc, newAxes);
    auto trueKeepdims = mlir::IntegerAttr::get(
        rewriter.getIntegerType(64, /*isSigned=*/true), 1);
    auto newReduce =
        rewriter.create<ONNX_OP>(loc, newReduceOutType, newReduceInput,
            newAxesVal, trueKeepdims, op.getNoopWithEmptyAxesAttr());

    // Trailing reshape iff canonical 4D shape differs from original.
    auto origOutShape = getShape(op.getReduced());
    mlir::Value result = newReduce.getReduced();
    if (newReduceOutShape != origOutShape)
      result = createReshape(rewriter, loc, result, origOutShape);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
}; // struct ReshapeReduceTo4DPattern

} // namespace

namespace onnx_mlir {

struct ReplaceQDQReductionPass
    : public mlir::PassWrapper<ReplaceQDQReductionPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const override {
    return "replace-qdq-reduction";
  }
  llvm::StringRef getDescription() const override {
    return "Reshape Reduce(Sum/Mean/Max/Min) to rank-4 + keep_dims=true.";
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    patterns.add<ReshapeReduceTo4DPattern<mlir::ONNXReduceSumOp>,
        ReshapeReduceTo4DPattern<mlir::ONNXReduceMeanOp>,
        ReshapeReduceTo4DPattern<mlir::ONNXReduceMaxOp>,
        ReshapeReduceTo4DPattern<mlir::ONNXReduceMinOp>>(context);

    mlir::GreedyRewriteConfig config;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createReplaceQDQReductionPass() {
  return std::make_unique<ReplaceQDQReductionPass>();
}

} // namespace onnx_mlir
