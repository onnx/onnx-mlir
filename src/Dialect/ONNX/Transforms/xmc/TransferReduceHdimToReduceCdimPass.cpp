// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

// ============================================================================
// TransferReduceHdimToReduceCdimPass
//
// Mirrors xcompiler's
// xcompiler-src/src/pass/passes/TransferReduceHdimToReduceCdimPass.cpp
//
// AIE's reduction kernel only runs on the last (channel / C-dim) axis.  This
// pass is the defensive cleanup that catches any onnx.ReduceSum / ReduceMean
// whose reduction axis is NOT the last dim, and rewrites it as
// `transpose -> reduce(axis = last) -> transpose` (or `reshape -> reduce` for
// the W-dim degenerate case).
//
// Two sub-patterns -- one each for the H-dim (rank-4) and the W-dim
// (rank-3 with degenerate channel) cases.  Output shape is preserved
// bit-exactly so consumers don't need to know anything changed.
//
// This pass should run AFTER TransferReduceMeanSumToConvPass (so we only
// shape reductions that didn't get conv-rewritten) and BEFORE any pass that
// folds adjacent transposes (so our inserted transposes stay isolated and
// can later be coalesced if appropriate).
// ============================================================================

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

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

/// Mirrors the helper in TransferReduceMeanSumToConvPass.cpp.  ONNX-MLIR's
/// transpose / reshape ops only accept float or quantized element types when
/// they participate in standard XMC lowering -- integer carriers (e.g. i64
/// axes/shape constants) should never reach here as the reduction's data.
bool isReduceCompatibleElementType(mlir::Type elementType) {
  return mlir::isa<mlir::FloatType>(elementType) ||
         mlir::isa<mlir::quant::QuantizedType>(elementType);
}

/// Extract axes from a ReduceSum/ReduceMean op's `axes` operand (must be a
/// constant).  Returns std::nullopt if the operand is missing or non-constant.
template <typename ONNX_OP>
std::optional<llvm::SmallVector<int64_t>> getConstantAxes(ONNX_OP op) {
  mlir::Value axesOperand = op.getAxes();
  if (!axesOperand)
    return std::nullopt;
  auto constOp = axesOperand.template getDefiningOp<mlir::ONNXConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto valueAttr = constOp.getValueAttr();
  if (!valueAttr)
    return std::nullopt;
  auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr);
  if (!denseAttr)
    return std::nullopt;
  llvm::SmallVector<int64_t> axes;
  for (auto v : denseAttr.getValues<mlir::APInt>())
    axes.push_back(v.getSExtValue());
  return axes;
}

/// Build a 1-D int64 constant tensor (used as axes / shape input).
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

/// Build an onnx.Transpose with the given perm.  Result shape is computed
/// from the input shape using perm.
mlir::Value createTranspose(mlir::PatternRewriter &rewriter,
    mlir::Location loc, mlir::Value input, llvm::ArrayRef<int64_t> perm) {
  auto inputType = mlir::cast<mlir::ShapedType>(input.getType());
  auto inputShape = inputType.getShape();
  llvm::SmallVector<int64_t> outputShape;
  outputShape.reserve(perm.size());
  for (int64_t p : perm)
    outputShape.push_back(inputShape[p]);
  auto outputType = mlir::RankedTensorType::get(
      outputShape, inputType.getElementType());
  auto permAttr = rewriter.getI64ArrayAttr(perm);
  return rewriter.create<mlir::ONNXTransposeOp>(
      loc, outputType, input, permAttr);
}

/// Build an onnx.Reshape with a constant shape input.  Element type
/// (incl. quantized) is carried through from the source value.
mlir::Value createReshape(mlir::PatternRewriter &rewriter, mlir::Location loc,
    mlir::Value input, llvm::ArrayRef<int64_t> targetShape) {
  auto inputType = mlir::cast<mlir::ShapedType>(input.getType());
  auto outputType = mlir::RankedTensorType::get(
      targetShape, inputType.getElementType());
  mlir::Value shapeConst = createInt64Const1D(rewriter, loc, targetShape);
  return rewriter.create<mlir::ONNXReshapeOp>(
      loc, outputType, input, shapeConst, /*allowzero=*/0);
}

//===----------------------------------------------------------------------===//
// ReduceHdimToCdimPattern (rank-4 NCHW H-dim → NHWC C-dim)
//
// Mirrors xcompiler::TransferReduceHdimToReduceCdimPass::transfer_reduce_h.
// For a rank-4 reduction with axis = [1] and keep_dims = true:
//
//   reshape  [N, C, H, W]  --transpose [0,2,3,1]-->  [N, H, W, C]
//                          --reduce axis=[3] keep_dims=true-->  [N, H, W, 1]
//                          --transpose [0,3,1,2]-->  [N, 1, H, W]
//
// Output type (`[N, 1, H, W]`) is preserved bit-exactly, so all downstream
// consumers see no change.
//===----------------------------------------------------------------------===//
template <typename ONNX_OP>
struct ReduceHdimToCdimPattern : public mlir::OpRewritePattern<ONNX_OP> {
  using mlir::OpRewritePattern<ONNX_OP>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ONNX_OP op, mlir::PatternRewriter &rewriter) const override {

    mlir::Value input = op.getData();
    auto inputShape = getShape(input);
    if (inputShape.size() != 4)
      return mlir::failure();

    auto inputElemType =
        mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
    if (!isReduceCompatibleElementType(inputElemType))
      return mlir::failure();

    // axes operand must be a constant single-element [1]
    auto axesOpt = getConstantAxes(op);
    if (!axesOpt || axesOpt->size() != 1)
      return mlir::failure();
    int64_t axis = normalizeAxis((*axesOpt)[0], 4);
    if (axis != 1)
      return mlir::failure();

    // keep_dims must be true (otherwise the trailing transpose's perm would
    // be wrong; keep_dims=false is left for other passes)
    if (op.getKeepdims() == 0)
      return mlir::failure();

    // Idempotency guard: if the input is already the output of our own
    // transpose [0,2,3,1], we'd loop forever.
    if (auto upTrans = input.template getDefiningOp<mlir::ONNXTransposeOp>()) {
      if (auto perm = upTrans.getPerm()) {
        llvm::SmallVector<int64_t> permVec;
        for (auto a : *perm)
          permVec.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());
        if (permVec == llvm::SmallVector<int64_t>{0, 2, 3, 1})
          return mlir::failure();
      }
    }

    mlir::Location loc = op.getLoc();

    // Step 1: transpose [0, 2, 3, 1]  → [N, H, W, C]
    mlir::Value transBefore =
        createTranspose(rewriter, loc, input, {0, 2, 3, 1});

    // Step 2: re-emit the reduction with axis = [3], keep_dims = true
    //         output shape: [N, H, W, 1]
    auto outputType = op.getReduced().getType();
    auto outputElemType =
        mlir::cast<mlir::ShapedType>(outputType).getElementType();
    llvm::SmallVector<int64_t> reduceOutShape = {
        inputShape[0], inputShape[2], inputShape[3], 1};
    auto reduceOutType =
        mlir::RankedTensorType::get(reduceOutShape, outputElemType);

    mlir::Value newAxes = createInt64Const1D(rewriter, loc, {3});

    auto newReduce = rewriter.create<ONNX_OP>(loc, reduceOutType, transBefore,
        newAxes, op.getKeepdimsAttr(), op.getNoopWithEmptyAxesAttr());

    // Step 3: transpose [0, 3, 1, 2]  → [N, 1, H, W]
    mlir::Value transAfter = createTranspose(
        rewriter, loc, newReduce.getReduced(), {0, 3, 1, 2});

    rewriter.replaceOp(op, transAfter);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ReduceWdimToCdimPattern (rank-3 axis=1 with degenerate channel)
//
// Mirrors xcompiler::TransferReduceHdimToReduceCdimPass::transfer_reduce_w.
// For a rank-3 reduction with axis = [1], keep_dims = true and shape[2] = 1:
//
//   reshape  [B, N, 1]  →  [B, 1, N]   (zero-cost: dim[2]==1 is a no-op view)
//   reduce   axis=[2] keep_dims=true  →  [B, 1, 1]
//
// Output type (`[B, 1, 1]`) is preserved bit-exactly.  No trailing transpose
// needed because the result frame already matches the original.
//===----------------------------------------------------------------------===//
template <typename ONNX_OP>
struct ReduceWdimToCdimPattern : public mlir::OpRewritePattern<ONNX_OP> {
  using mlir::OpRewritePattern<ONNX_OP>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ONNX_OP op, mlir::PatternRewriter &rewriter) const override {

    mlir::Value input = op.getData();
    auto inputShape = getShape(input);
    if (inputShape.size() != 3)
      return mlir::failure();

    auto inputElemType =
        mlir::cast<mlir::ShapedType>(input.getType()).getElementType();
    if (!isReduceCompatibleElementType(inputElemType))
      return mlir::failure();

    auto axesOpt = getConstantAxes(op);
    if (!axesOpt || axesOpt->size() != 1)
      return mlir::failure();
    int64_t axis = normalizeAxis((*axesOpt)[0], 3);
    if (axis != 1)
      return mlir::failure();

    if (op.getKeepdims() == 0)
      return mlir::failure();

    // Channel position must be degenerate (dim[2] == 1)
    if (inputShape[2] != 1)
      return mlir::failure();
    // Don't fire if the reduction is already a no-op (dim[1] == 1)
    if (inputShape[1] == 1)
      return mlir::failure();

    mlir::Location loc = op.getLoc();

    // Step 1: reshape [B, N, 1] → [B, 1, N]
    llvm::SmallVector<int64_t> reshapedShape = {
        inputShape[0], 1, inputShape[1]};
    mlir::Value reshaped = createReshape(rewriter, loc, input, reshapedShape);

    // Step 2: re-emit the reduction with axis = [2]
    //         output shape preserved (= original output shape [B,1,1])
    auto outputType = op.getReduced().getType();
    mlir::Value newAxes = createInt64Const1D(rewriter, loc, {2});

    auto newReduce = rewriter.create<ONNX_OP>(loc, outputType, reshaped,
        newAxes, op.getKeepdimsAttr(), op.getNoopWithEmptyAxesAttr());

    rewriter.replaceOp(op, newReduce.getReduced());
    return mlir::success();
  }
};


//===----------------------------------------------------------------------===//
// PadReduceTo4DPattern
//
// Mirrors xcompiler's ReplaceQDQReductionPass::shape_to_4d for the rank<4 →
// rank-4 promotion needed by the AIE reduction kernel.  For an
// ONNX_OP whose input is rank-1, rank-2 or rank-3, this pattern:
//
//   1. Inserts a leading-1 reshape to promote the input to rank-4
//   2. Shifts the axis attribute by (4 - rank) so it points at the same
//      logical dim
//   3. Flips keepdims=false to true so the new reduce stays rank-4
//   4. Appends a trailing reshape to restore the original output shape
//
// After this pattern fires, the existing ReduceHdimToCdimPattern can fire on
// the now-rank-4 reduce (when the shifted axis lands on 1) and emit the
// transpose-sandwich, producing the same op chain xmodel emits.
//
// Match preconditions (kept narrow to avoid affecting unrelated reductions):
//   - rank-1, rank-2, or rank-3 input (rank-4 already handled directly)
//   - single axis (axis array size == 1)
//   - axes operand is a static ONNXConstantOp
//   - input element type is float or quantized (i64/i32 reductions need
//     special Cast handling and are out of scope here)
//   - input has static shape
//   - the shifted axis lands on position 1 (NCHW H-dim) so a downstream
//     ReduceHdimToCdimPattern can fire; otherwise we don't pad
//   - input is not already a leading-1 reshape we inserted (idempotency)
//===----------------------------------------------------------------------===//
template <typename ONNX_OP>
struct PadReduceTo4DPattern : public mlir::OpRewritePattern<ONNX_OP> {
  using mlir::OpRewritePattern<ONNX_OP>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ONNX_OP op, mlir::PatternRewriter &rewriter) const override {

    mlir::Value input = op.getData();
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape())
      return mlir::failure();
    auto inputShape = getShape(input);
    int64_t rank = inputShape.size();

    if (rank >= 4 || rank < 1)
      return mlir::failure();

    if (!isReduceCompatibleElementType(inputType.getElementType()))
      return mlir::failure();

    auto axesOpt = getConstantAxes(op);
    if (!axesOpt || axesOpt->size() != 1)
      return mlir::failure();
    int64_t axis = normalizeAxis((*axesOpt)[0], rank);
    int64_t shift = 4 - rank;
    int64_t newAxis = axis + shift;

    // Only pad if the shifted axis lands on a position where the downstream
    // shaping patterns can take over.  Currently that's axis==1 (matches
    // ReduceHdimToCdimPattern after padding).
    if (newAxis != 1)
      return mlir::failure();

    // Idempotency: skip if input is a leading-1-prepending reshape we already
    // inserted (this would loop us forever).
    if (auto upReshape =
            input.template getDefiningOp<mlir::ONNXReshapeOp>()) {
      auto inT = mlir::dyn_cast<mlir::RankedTensorType>(
          upReshape.getData().getType());
      auto outT = mlir::dyn_cast<mlir::RankedTensorType>(
          upReshape.getReshaped().getType());
      if (inT && outT && outT.getRank() == 4 && inT.getRank() < 4 &&
          outT.getShape()[0] == 1)
        return mlir::failure();
    }

    mlir::Location loc = op.getLoc();
    auto outElemType =
        mlir::cast<mlir::ShapedType>(op.getReduced().getType())
            .getElementType();

    // Step 1: pad input to rank-4 by prepending size-1 dims
    llvm::SmallVector<int64_t> paddedShape(shift, 1);
    paddedShape.insert(paddedShape.end(), inputShape.begin(), inputShape.end());
    mlir::Value padded = createReshape(rewriter, loc, input, paddedShape);

    // Step 2: re-emit the reduce with axis = [newAxis], keepdims = true
    llvm::SmallVector<int64_t> reduceOutShape(paddedShape);
    reduceOutShape[newAxis] = 1;
    auto reduceOutType =
        mlir::RankedTensorType::get(reduceOutShape, outElemType);

    mlir::Value newAxes = createInt64Const1D(rewriter, loc, {newAxis});

    auto trueKeepdims = mlir::IntegerAttr::get(
        rewriter.getIntegerType(64, /*isSigned=*/true), 1);
    auto newReduce = rewriter.create<ONNX_OP>(loc, reduceOutType, padded,
        newAxes, trueKeepdims, op.getNoopWithEmptyAxesAttr());

    // Step 3: trailing reshape to restore original output shape
    auto origOutShape = getShape(op.getReduced());
    mlir::Value result = createReshape(rewriter, loc,
        newReduce.getReduced(), origOutShape);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
}; // struct PadReduceTo4DPattern

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

struct TransferReduceHdimToReduceCdimPass
    : public mlir::PassWrapper<TransferReduceHdimToReduceCdimPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
  llvm::StringRef getArgument() const override {
    return "transfer-reduce-hdim-to-reduce-cdim";
  }
  llvm::StringRef getDescription() const override {
    return "Transfer ReduceSum/ReduceMean on H/W-dim to a C-dim reduction by "
           "inserting transposes/reshapes around the reduction.  Mirrors "
           "xcompiler's TransferReduceHdimToReduceCdimPass.";
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    // ReduceSum + ReduceMean (newer ONNX versions, axes-as-input form).
    // Older V11/V13 variants store axes as an attribute and would need a
    // separate (similarly templated) pattern -- omitted here because the
    // companion TransferReduceMeanSumToConvPass also only handles the
    // newer ops.
    patterns.add<PadReduceTo4DPattern<mlir::ONNXReduceSumOp>>(context);
    patterns.add<PadReduceTo4DPattern<mlir::ONNXReduceMeanOp>>(context);
    patterns.add<ReduceHdimToCdimPattern<mlir::ONNXReduceSumOp>>(context);
    patterns.add<ReduceWdimToCdimPattern<mlir::ONNXReduceSumOp>>(context);
    patterns.add<ReduceHdimToCdimPattern<mlir::ONNXReduceMeanOp>>(context);
    patterns.add<ReduceWdimToCdimPattern<mlir::ONNXReduceMeanOp>>(context);

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

std::unique_ptr<mlir::Pass> createTransferReduceHdimToReduceCdimPass() {
  return std::make_unique<TransferReduceHdimToReduceCdimPass>();
}

} // namespace onnx_mlir
