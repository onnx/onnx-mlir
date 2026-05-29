// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// PropagateQuantTypeThroughDataFlowPass
//
// Post-`quant-types` pass. For data-flow ops (no computation per value; every
// output value equals some input value, all sharing the same quantization
// parameters), unifies an f32 <-> !quant.uniform mismatch between the data
// operand(s) and result(s) by retyping the f32 side to the quant type in place
// via Value::setType.
//
// At function input/output boundaries the pass DOES NOT propagate. If a data
// operand is a function block argument, or any user of a data result is a
// `func.return`, that side is treated as a hard skip — no quant type transfer
// occurs there and no bridging Q/scast/DQ ops are inserted. The function ABI
// stays untouched.
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

// Whitelist matches xcompiler's QDQStandardizeQuantizationPass plus basic
// shape-only ops that are trivial extensions of Reshape.
static bool isWhitelistedDataFlow(Operation *op) {
  return isa<mlir::ONNXReshapeOp, mlir::ONNXTransposeOp, mlir::ONNXSqueezeOp,
      mlir::ONNXSqueezeV11Op, mlir::ONNXUnsqueezeOp, mlir::ONNXUnsqueezeV11Op,
      mlir::ONNXFlattenOp, mlir::ONNXIdentityOp, mlir::ONNXDepthToSpaceOp,
      mlir::ONNXSpaceToDepthOp, mlir::ONNXReverseSequenceOp, mlir::ONNXExpandOp,
      mlir::ONNXNegOp, mlir::ONNXClipOp, mlir::ONNXResizeOp, mlir::ONNXConcatOp,
      mlir::ONNXWhereOp>(op);
}

// Attribute-filter gate. Resize is only safe for nearest mode.
static bool passesAttributeFilter(Operation *op) {
  if (auto resize = dyn_cast<mlir::ONNXResizeOp>(op))
    return resize.getMode() == "nearest";
  return true;
}

// Returns the data operand and data result Values for `op`. Metadata operands
// (shape / axes / indices / pads / constant_value / scales / sizes / repeats /
// condition / sequence_lens / roi) and metadata results are excluded.
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

// Consumers whose f32 operand type is semantically bound. Includes
// func.return: at function output boundary we never retype, never bridge.
static bool isTypeBoundConsumer(Operation *op) {
  return isa<mlir::ONNXCastOp, mlir::ONNXCastLikeOp, mlir::ONNXQuantizeLinearOp,
      func::ReturnOp>(op);
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
      // Skip if any user of any result is type-bound. This includes func.return
      // (graph output boundary) and Cast/CastLike/QuantizeLinear.
      for (Value r : io.results)
        for (Operation *u : r.getUsers())
          if (isTypeBoundConsumer(u))
            return failure();
      rewriter.modifyOpInPlace(op, [&]() {
        for (Value r : io.results) {
          Type newTy = cast<TensorType>(r.getType()).clone(operandQ);
          r.setType(newTy);
        }
      });
      return success();
    }

    // Backward: at least one data operand is f32, all data results share a
    // common quant type. Retype per-operand: skip operands at the function
    // input boundary (block-arg) or with type-bound siblings; retype the rest
    // via in-place setType on the producer.
    if (resultQ && anyOperandF32) {
      SmallVector<bool> retypeThis(io.operands.size(), false);
      SmallVector<Operation *> producers(io.operands.size(), nullptr);
      for (auto [i, opnd] : llvm::enumerate(io.operands)) {
        if (!isF32Tensor(opnd.getType()))
          continue; // already quant
        Operation *p = opnd.getDefiningOp();
        if (!p)
          continue; // function block argument: at boundary, skip
        if (isTypeBoundProducer(p))
          continue; // hard skip for type-bound producer
        // Check siblings of this operand. If any is a type-bound consumer
        // (Cast/CastLike/Q/func.return), skip retyping this operand.
        bool hasBadSibling = false;
        for (Operation *u : opnd.getUsers()) {
          if (u == op)
            continue;
          if (isTypeBoundConsumer(u)) {
            hasBadSibling = true;
            break;
          }
        }
        if (hasBadSibling)
          continue;
        retypeThis[i] = true;
        producers[i] = p;
      }
      // Nothing to retype? Bail out.
      if (!llvm::any_of(retypeThis, [](bool b) { return b; }))
        return failure();
      for (auto [i, opnd] : llvm::enumerate(io.operands)) {
        if (!retypeThis[i])
          continue;
        Operation *p = producers[i];
        Type newTy = cast<TensorType>(opnd.getType()).clone(resultQ);
        rewriter.modifyOpInPlace(p, [&]() { opnd.setType(newTy); });
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
           "Function input and output boundaries are skipped (no Q/DQ/scast "
           "bridging at the ABI).";
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
