// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// PropagateQuantTypeThroughDataFlowPass
//
// Post-`quant-types` pass. For data-flow ops (no computation per value; every
// output value equals some input value, all sharing the same quantization
// parameters), unifies an f32 <-> !quant.uniform mismatch between the data
// operand(s) and result(s) by retyping the f32 side to the quant type in place
// via Value::setType. At function input/output boundaries an
// ONNXQuantizeLinear+scast or scast+ONNXDequantizeLinear bridge is inserted so
// the ABI stays f32.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace {

static bool isF32Tensor(Type t) {
  auto tt = dyn_cast<TensorType>(t);
  return tt && tt.getElementType().isF32();
}

static quant::QuantizedType getQuantElement(Type t) {
  auto tt = dyn_cast<TensorType>(t);
  if (!tt)
    return nullptr;
  return dyn_cast<quant::QuantizedType>(tt.getElementType());
}

// Returns the common quantized element type across `values`, or null if any
// value is non-quant or two values disagree.
static quant::QuantizedType getCommonQuant(ArrayRef<Value> values) {
  quant::QuantizedType common;
  for (Value v : values) {
    auto q = getQuantElement(v.getType());
    if (!q)
      return nullptr;
    if (!common)
      common = q;
    else if (common != q)
      return nullptr;
  }
  return common;
}

static bool allF32Tensors(ArrayRef<Value> values) {
  for (Value v : values)
    if (!isF32Tensor(v.getType()))
      return false;
  return !values.empty();
}

// True if `op` is a data-flow op (every output value equals some input value
// with the same quant params). The data operands and results may be a subset
// of all operands/results; see getDataIO().
//
// Whitelist matches xcompiler's QDQStandardizeQuantizationPass:
//   {transpose, reshape, pixel-shuffle, expand, neg, clamp, concat, resize,
//    where}
// plus basic shape-only ops (Squeeze/Unsqueeze/Flatten/Identity/
// ReverseSequence/SpaceToDepth) which are trivial extensions of Reshape that
// some models don't pre-decompose.
static bool isWhitelistedDataFlow(Operation *op) {
  return isa<
      // Basic shape-only / no-computation (same element count, rearrangement)
      mlir::ONNXReshapeOp, mlir::ONNXTransposeOp, mlir::ONNXSqueezeOp,
      mlir::ONNXSqueezeV11Op, mlir::ONNXUnsqueezeOp, mlir::ONNXUnsqueezeV11Op,
      mlir::ONNXFlattenOp, mlir::ONNXIdentityOp, mlir::ONNXDepthToSpaceOp,
      mlir::ONNXSpaceToDepthOp, mlir::ONNXReverseSequenceOp,
      // QDQStandardize: value-preserving / accepted-approximation single-input
      mlir::ONNXExpandOp, mlir::ONNXNegOp, mlir::ONNXClipOp,
      // QDQStandardize: single input, attribute-filtered
      mlir::ONNXResizeOp,
      // QDQStandardize: multi-input
      mlir::ONNXConcatOp, mlir::ONNXWhereOp>(op);
}

// Attribute-filter gate. Returns true if `op` either has no filter or passes
// the filter (currently: Resize only when mode=="nearest").
static bool passesAttributeFilter(Operation *op) {
  if (auto resize = dyn_cast<mlir::ONNXResizeOp>(op))
    return resize.getMode() == "nearest";
  return true;
}

// Returns the data operand and data result Values for `op`. Metadata operands
// (shape / axes / indices / pads / constant_value / scales / sizes / repeats /
// condition / sequence_lens / roi) and metadata results (indices) are
// excluded.
struct DataIO {
  SmallVector<Value> operands;
  SmallVector<Value> results;
};

static DataIO getDataIO(Operation *op) {
  DataIO io;
  if (auto concat = dyn_cast<mlir::ONNXConcatOp>(op)) {
    for (Value v : concat.getInputs())
      io.operands.push_back(v);
    io.results.push_back(concat.getConcatResult());
    return io;
  }
  if (auto where = dyn_cast<mlir::ONNXWhereOp>(op)) {
    // Operands: condition (i1), X, Y. Only X and Y carry data.
    io.operands.push_back(where.getX());
    io.operands.push_back(where.getY());
    io.results.push_back(where.getResult());
    return io;
  }
  // Default: single primary data operand at index 0; single data result at 0.
  if (op->getNumOperands() < 1 || op->getNumResults() < 1)
    return io;
  io.operands.push_back(op->getOperand(0));
  io.results.push_back(op->getResult(0));
  return io;
}

// Producers whose f32 result element type is bound to attributes/region/operand
// and cannot be retyped by Value::setType.
static bool isTypeBoundProducer(Operation *op) {
  return isa<mlir::ONNXConstantOp, mlir::ONNXConstantOfShapeOp,
      mlir::ONNXCastOp, mlir::ONNXCastLikeOp, mlir::ONNXRangeOp,
      mlir::ONNXEyeLikeOp, mlir::ONNXRandomNormalOp,
      mlir::ONNXRandomNormalLikeOp, mlir::ONNXRandomUniformOp,
      mlir::ONNXRandomUniformLikeOp, mlir::ONNXMultinomialOp,
      mlir::ONNXBernoulliOp, mlir::ONNXDequantizeLinearOp, mlir::ONNXIfOp,
      mlir::ONNXLoopOp, mlir::ONNXScanOp>(op);
}

// Consumers whose f32 operand type is interpreted literally (Cast/CastLike)
// or paired with the op's own quantization attributes (Q).
static bool isTypeBoundConsumer(Operation *op) {
  return isa<mlir::ONNXCastOp, mlir::ONNXCastLikeOp,
      mlir::ONNXQuantizeLinearOp>(op);
}

static std::pair<Value, Value> buildScaleZpConstants(
    PatternRewriter &rewriter, Location loc, quant::UniformQuantizedType q) {
  auto scaleTT = RankedTensorType::get({}, rewriter.getF32Type());
  auto scaleAttr =
      DenseElementsAttr::get(scaleTT, rewriter.getF32FloatAttr(q.getScale()));
  auto scaleC =
      rewriter.create<mlir::ONNXConstantOp>(loc, Attribute(), scaleAttr);
  Type storageTy = q.getStorageType();
  auto zpTT = RankedTensorType::get({}, storageTy);
  auto zpAttr = DenseElementsAttr::get(
      zpTT, rewriter.getIntegerAttr(storageTy, q.getZeroPoint()));
  auto zpC = rewriter.create<mlir::ONNXConstantOp>(loc, Attribute(), zpAttr);
  return {scaleC.getResult(), zpC.getResult()};
}

// f32Val -> ONNXQuantizeLinear -> quant.scast -> returned !quant.uniform value.
static Value insertF32ToQuantBridge(PatternRewriter &rewriter, Location loc,
    Value f32Val, quant::UniformQuantizedType q) {
  auto f32TT = cast<TensorType>(f32Val.getType());
  auto [scaleC, zpC] = buildScaleZpConstants(rewriter, loc, q);
  auto qOp = rewriter.create<mlir::ONNXQuantizeLinearOp>(loc,
      f32TT.clone(q.getStorageType()), f32Val, scaleC, zpC,
      /*axis=*/IntegerAttr(), /*saturate=*/IntegerAttr(),
      /*block_size=*/IntegerAttr());
  return rewriter.create<quant::StorageCastOp>(loc, f32TT.clone(q), qOp)
      .getResult();
}

// quantVal -> quant.scast -> ONNXDequantizeLinear -> returned f32 value.
static Value insertQuantToF32Bridge(PatternRewriter &rewriter, Location loc,
    Value quantVal, quant::UniformQuantizedType q) {
  auto qTT = cast<TensorType>(quantVal.getType());
  auto scast = rewriter.create<quant::StorageCastOp>(
      loc, qTT.clone(q.getStorageType()), quantVal);
  auto [scaleC, zpC] = buildScaleZpConstants(rewriter, loc, q);
  return rewriter
      .create<mlir::ONNXDequantizeLinearOp>(loc,
          qTT.clone(rewriter.getF32Type()), scast.getResult(), scaleC, zpC,
          /*axis=*/IntegerAttr(), /*block_size=*/IntegerAttr())
      .getResult();
}

// Examines users of `v`. Collects `func.return` users into `returnUsers`;
// returns failure() if any non-return user is type-bound.
static LogicalResult classifyUsers(
    Value v, SmallVector<func::ReturnOp> &returnUsers) {
  for (Operation *u : v.getUsers()) {
    if (auto r = dyn_cast<func::ReturnOp>(u)) {
      returnUsers.push_back(r);
      continue;
    }
    if (isTypeBoundConsumer(u))
      return failure();
  }
  return success();
}

// Examines sibling users of `v` (users that are not `self`). Collects return
// siblings; fails on any non-return type-bound sibling.
static LogicalResult classifySiblings(
    Value v, Operation *self, SmallVector<func::ReturnOp> &returnSiblings) {
  for (Operation *u : v.getUsers()) {
    if (u == self)
      continue;
    if (auto r = dyn_cast<func::ReturnOp>(u)) {
      returnSiblings.push_back(r);
      continue;
    }
    if (isTypeBoundConsumer(u))
      return failure();
  }
  return success();
}

struct PropagateQuantTypePattern : public RewritePattern {
  PropagateQuantTypePattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!isWhitelistedDataFlow(op))
      return failure();
    if (!passesAttributeFilter(op))
      return failure();

    DataIO io = getDataIO(op);
    if (io.operands.empty() || io.results.empty())
      return failure();

    // Per-axis is weights-only, not on the activation path here.
    auto isPerAxis = [](quant::QuantizedType q) {
      return q && isa<quant::UniformQuantizedPerAxisType>(q);
    };
    for (Value v : io.operands)
      if (isPerAxis(getQuantElement(v.getType())))
        return failure();
    for (Value v : io.results)
      if (isPerAxis(getQuantElement(v.getType())))
        return failure();

    auto operandQ = getCommonQuant(io.operands);
    auto resultQ = getCommonQuant(io.results);
    bool resultsF32 = allF32Tensors(io.results);
    bool anyOperandF32 = llvm::any_of(
        io.operands, [](Value v) { return isF32Tensor(v.getType()); });

    // Forward: data operands share a common quant type, results are all f32.
    if (operandQ && resultsF32) {
      auto uQ = dyn_cast<quant::UniformQuantizedType>(operandQ);
      SmallVector<SmallVector<func::ReturnOp>> resultReturns(io.results.size());
      for (auto [i, r] : llvm::enumerate(io.results)) {
        if (failed(classifyUsers(r, resultReturns[i])))
          return failure();
      }
      // Function-return bridges require a per-tensor uniform quant.
      for (auto &rus : resultReturns)
        if (!rus.empty() && !uQ)
          return failure();
      // Retype all data results in place.
      rewriter.modifyOpInPlace(op, [&]() {
        for (Value r : io.results) {
          Type newTy = cast<TensorType>(r.getType()).clone(operandQ);
          r.setType(newTy);
        }
      });
      // Bridge func.return users back to f32.
      for (auto [i, r] : llvm::enumerate(io.results)) {
        for (func::ReturnOp ret : resultReturns[i]) {
          rewriter.setInsertionPoint(ret);
          Value bridge = insertQuantToF32Bridge(rewriter, op->getLoc(), r, uQ);
          rewriter.modifyOpInPlace(ret, [&]() {
            for (OpOperand &opnd : ret->getOpOperands())
              if (opnd.get() == r)
                opnd.set(bridge);
          });
        }
      }
      return success();
    }

    // Backward: data results share a common quant type, AT LEAST one data
    // operand is f32 (matching xcompiler's case 2 "all-missing" and case 3
    // "partial-missing" together). Already-quant operands are left untouched
    // even if they disagree with resultQ -- their pre-pass mismatch state is
    // preserved (no worse than skipping the rewrite entirely).
    if (resultQ && anyOperandF32) {
      auto uQ = dyn_cast<quant::UniformQuantizedType>(resultQ);
      // Classify each f32 operand independently. Any single classification
      // failure aborts the whole rewrite so we don't half-apply.
      SmallVector<bool> retypeThis(io.operands.size(), false);
      SmallVector<bool> isBlockArg(io.operands.size(), false);
      SmallVector<Operation *> producers(io.operands.size(), nullptr);
      SmallVector<SmallVector<func::ReturnOp>> operandReturnSiblings(
          io.operands.size());
      for (auto [i, opnd] : llvm::enumerate(io.operands)) {
        if (!isF32Tensor(opnd.getType()))
          continue; // skip already-quant operand; leave it as-is
        retypeThis[i] = true;
        if (isa<BlockArgument>(opnd)) {
          if (!uQ)
            return failure();
          isBlockArg[i] = true;
          continue;
        }
        Operation *p = opnd.getDefiningOp();
        if (!p || isTypeBoundProducer(p))
          return failure();
        if (failed(classifySiblings(opnd, op, operandReturnSiblings[i])))
          return failure();
        if (!operandReturnSiblings[i].empty() && !uQ)
          return failure();
        producers[i] = p;
      }
      // Apply: for block-arg operands, insert local Q+scast bridge; for
      // regular operands, retype the producer's result in place. Skip
      // already-quant operands entirely.
      for (auto [i, opnd] : llvm::enumerate(io.operands)) {
        if (!retypeThis[i])
          continue;
        if (isBlockArg[i]) {
          rewriter.setInsertionPoint(op);
          Value bridge =
              insertF32ToQuantBridge(rewriter, op->getLoc(), opnd, uQ);
          unsigned operandIdx = 0;
          for (OpOperand &use : op->getOpOperands()) {
            if (use.get() == opnd && operandIdx == i) {
              use.set(bridge);
              break;
            }
            if (use.get() == opnd)
              ++operandIdx;
          }
        } else {
          Operation *p = producers[i];
          Type newTy = cast<TensorType>(opnd.getType()).clone(resultQ);
          rewriter.modifyOpInPlace(p, [&]() { opnd.setType(newTy); });
          for (func::ReturnOp ret : operandReturnSiblings[i]) {
            rewriter.setInsertionPoint(ret);
            Value bridge =
                insertQuantToF32Bridge(rewriter, op->getLoc(), opnd, uQ);
            rewriter.modifyOpInPlace(ret, [&]() {
              for (OpOperand &use : ret->getOpOperands())
                if (use.get() == opnd)
                  use.set(bridge);
            });
          }
        }
      }
      return success();
    }

    return failure();
  }
};

} // namespace

namespace onnx_mlir {

struct PropagateQuantTypeThroughDataFlowPass
    : public PassWrapper<PropagateQuantTypeThroughDataFlowPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "propagate-quant-type-through-dataflow";
  }
  StringRef getDescription() const override {
    return "Unify f32 <-> !quant.uniform across data-flow ops matching "
           "xcompiler's QDQStandardizeQuantizationPass: Reshape/Transpose/"
           "Squeeze/Unsqueeze/Flatten/Identity/DepthToSpace/SpaceToDepth/"
           "ReverseSequence/Expand/Neg/Clip/Resize(nearest)/Concat/Where. "
           "Function inputs and outputs are bridged with QuantizeLinear+"
           "scast or scast+DequantizeLinear so the ABI stays f32.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quant::QuantDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PropagateQuantTypePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createPropagateQuantTypeThroughDataFlowPass() {
  return std::make_unique<PropagateQuantTypeThroughDataFlowPass>();
}

} // namespace onnx_mlir
