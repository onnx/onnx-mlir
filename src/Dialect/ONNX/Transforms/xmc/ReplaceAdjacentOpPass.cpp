// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <atomic>

using namespace mlir;

namespace {

/// Pattern to merge nested concat operations into a single concat.
/// Pattern: concat(concat(a,b), concat(c,d), e) -> concat(a,b,c,d,e)
struct MergeNestedConcatPattern : public OpRewritePattern<ONNXConcatOp> {
  using OpRewritePattern<ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXConcatOp concatOp,
      PatternRewriter &rewriter) const override {
    auto axisAttr = concatOp.getAxis();
    if (!axisAttr)
      return rewriter.notifyMatchFailure(concatOp, "No axis attribute");

    int64_t axis = axisAttr;
    auto outputType = dyn_cast<RankedTensorType>(concatOp.getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(
          concatOp, "Output is not a ranked tensor");

    // Normalize negative axis.
    if (axis < 0)
      axis += outputType.getRank();

    // If quantized, require matching quant params.
    auto outputQType =
        dyn_cast<quant::UniformQuantizedType>(outputType.getElementType());

    llvm::SmallVector<Value> flattenedInputs;
    bool foundMergeableConcat = false;

    for (auto input : concatOp.getInputs()) {
      auto inputConcatOp = input.getDefiningOp<ONNXConcatOp>();

      // Check if input is a concat that can be merged.
      if (inputConcatOp && inputConcatOp->hasOneUse()) {
        auto inputAxisAttr = inputConcatOp.getAxis();
        if (!inputAxisAttr) {
          flattenedInputs.push_back(input);
          continue;
        }

        int64_t inputAxis = inputAxisAttr;
        auto inputOutputType =
            dyn_cast<RankedTensorType>(inputConcatOp.getType());
        if (inputOutputType && inputAxis < 0)
          inputAxis += inputOutputType.getRank();

        if (inputAxis != axis) {
          flattenedInputs.push_back(input);
          continue;
        }

        // Check quantization parameters match if quantized.
        if (outputQType) {
          auto inputQType = dyn_cast<quant::UniformQuantizedType>(
              inputOutputType.getElementType());
          if (!inputQType || inputQType.getScale() != outputQType.getScale() ||
              inputQType.getZeroPoint() != outputQType.getZeroPoint()) {
            flattenedInputs.push_back(input);
            continue;
          }
        }

        // Check all nested inputs' quantization before merging.
        bool canMergeNested = true;
        if (outputQType) {
          for (auto nestedInput : inputConcatOp.getInputs()) {
            auto nestedInputType =
                dyn_cast<RankedTensorType>(nestedInput.getType());
            if (!nestedInputType)
              continue;
            auto nestedInputQType = dyn_cast<quant::UniformQuantizedType>(
                nestedInputType.getElementType());
            if (!nestedInputQType ||
                nestedInputQType.getScale() != outputQType.getScale() ||
                nestedInputQType.getZeroPoint() != outputQType.getZeroPoint()) {
              canMergeNested = false;
              break;
            }
          }
        }

        if (canMergeNested) {
          foundMergeableConcat = true;
          for (auto nestedInput : inputConcatOp.getInputs())
            flattenedInputs.push_back(nestedInput);
        } else {
          flattenedInputs.push_back(input);
        }
      } else {
        flattenedInputs.push_back(input);
      }
    }

    if (!foundMergeableConcat) {
      return rewriter.notifyMatchFailure(
          concatOp, "No mergeable nested concat found");
    }

    rewriter.setInsertionPoint(concatOp);
    auto newConcatOp = rewriter.create<ONNXConcatOp>(concatOp.getLoc(),
        concatOp.getType(), flattenedInputs, concatOp.getAxisAttr());

    rewriter.replaceOp(concatOp, newConcatOp.getResult());
    return success();
  }
};

/// Create a no-op onnx.Reshape inserted after the producer (or block start for
/// block arguments) to create a distinct SSA value for duplicate uses.
static Value insertIdentityReshapeAfterProducer(
    PatternRewriter &rewriter, Value input) {
  Operation *definingOp = input.getDefiningOp();
  Location loc = input.getLoc();

  if (definingOp) {
    rewriter.setInsertionPointAfter(definingOp);
  } else {
    auto blockArg = cast<BlockArgument>(input);
    rewriter.setInsertionPointToStart(blockArg.getOwner());
  }

  auto inputType = input.getType();
  if (auto tensorType = dyn_cast<TensorType>(inputType)) {
    auto shapeType =
        RankedTensorType::get({ShapedType::kDynamic}, rewriter.getI64Type());

    OperationState shapeState(loc, "onnx.Shape");
    shapeState.addOperands(input);
    shapeState.addTypes(shapeType);
    Operation *shapeOp = rewriter.create(shapeState);

    OperationState reshapeState(loc, "onnx.Reshape");
    reshapeState.addOperands({input, shapeOp->getResult(0)});
    reshapeState.addTypes(tensorType);
    auto si64Ty = rewriter.getIntegerType(64, IntegerType::Signed);
    reshapeState.addAttribute("allowzero", rewriter.getIntegerAttr(si64Ty, 0));
    Operation *reshapeOp = rewriter.create(reshapeState);

    // This transformation exists to break duplicate SSA uses for backends that
    // require distinct input edges (XCompiler parity). Guard against later CSE
    // merging the inserted no-op reshapes back together by giving each inserted
    // reshape a unique id.
    static std::atomic<int64_t> nextId{0};
    reshapeOp->setAttr("duplicate_input", rewriter.getBoolAttr(true));
    reshapeOp->setAttr(
        "duplicate_input_id", rewriter.getI64IntegerAttr(nextId++));
    return reshapeOp->getResult(0);
  }

  return Value();
}

/// Split duplicate inputs by inserting no-op reshapes.
struct SplitDuplicateInputsPattern : public RewritePattern {
  SplitDuplicateInputsPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 0 || op->hasTrait<OpTrait::IsTerminator>())
      return failure();

    llvm::DenseSet<Value> uniqueInputs;
    llvm::DenseMap<Value, unsigned> duplicateCount;

    for (Value operand : op->getOperands()) {
      uniqueInputs.insert(operand);
      duplicateCount[operand]++;
    }

    if (uniqueInputs.size() == op->getNumOperands())
      return failure();

    llvm::DenseMap<Value, SmallVector<Value, 4>> inputMap;
    for (Value uniqueInput : uniqueInputs) {
      auto count = duplicateCount[uniqueInput];
      auto &valueList = inputMap[uniqueInput];

      valueList.push_back(uniqueInput);
      for (unsigned i = 1; i < count; i++) {
        Value reshapeResult = insertIdentityReshapeAfterProducer(rewriter,
            uniqueInput);
        if (!reshapeResult)
          return failure();
        valueList.push_back(reshapeResult);
      }
    }

    SmallVector<Value, 4> orderedInputs;
    for (Value originalOperand : op->getOperands()) {
      auto &candidates = inputMap[originalOperand];
      if (candidates.empty())
        return rewriter.notifyMatchFailure(op,
            "internal error: missing operand candidate while splitting "
            "duplicates");
      orderedInputs.push_back(candidates.front());
      candidates.erase(candidates.begin());
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(orderedInputs);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());

    rewriter.setInsertionPoint(op);
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct ReplaceAdjacentOpPass
    : public PassWrapper<ReplaceAdjacentOpPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "replace-adjacent-ops"; }
  StringRef getDescription() const override {
    return "Merge nested concat operations and split duplicate operand uses by "
           "inserting identity reshapes";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<MergeNestedConcatPattern>(ctx);
    patterns.add<SplitDuplicateInputsPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createReplaceAdjacentOpPass() {
  return std::make_unique<ReplaceAdjacentOpPass>();
}

} // namespace onnx_mlir

