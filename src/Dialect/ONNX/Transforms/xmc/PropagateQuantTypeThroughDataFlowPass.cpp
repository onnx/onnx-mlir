// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// PropagateQuantTypeThroughDataFlowPass
//
// Post-`quant-types` pass. For pure data-flow ops (no computation, same
// element count) with an f32 <-> !quant.uniform mismatch between the data
// operand and result, the f32 side is retyped in place to the quant type.
// At graph (function) input/output boundaries an ONNXQuantizeLinear+scast or
// scast+ONNXDequantizeLinear bridge is inserted so the function ABI stays f32.
// All other type-bound boundaries are left untouched.
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

static bool isWhitelistedDataFlow(Operation *op) {
  return isa<mlir::ONNXReshapeOp, mlir::ONNXTransposeOp, mlir::ONNXSqueezeOp,
      mlir::ONNXSqueezeV11Op, mlir::ONNXUnsqueezeOp, mlir::ONNXUnsqueezeV11Op,
      mlir::ONNXFlattenOp, mlir::ONNXIdentityOp, mlir::ONNXDepthToSpaceOp,
      mlir::ONNXSpaceToDepthOp, mlir::ONNXReverseSequenceOp>(op);
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

struct PropagateQuantTypePattern : public RewritePattern {
  PropagateQuantTypePattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!isWhitelistedDataFlow(op))
      return failure();

    Value in = op->getOperand(0);
    Value out = op->getResult(0);
    Type inET = cast<TensorType>(in.getType()).getElementType();
    Type outET = cast<TensorType>(out.getType()).getElementType();
    if (inET == outET)
      return failure();

    auto inQ = dyn_cast<quant::QuantizedType>(inET);
    auto outQ = dyn_cast<quant::QuantizedType>(outET);

    // Per-axis is weights-only, not on the activation path here.
    if (inQ && isa<quant::UniformQuantizedPerAxisType>(inQ))
      return failure();
    if (outQ && isa<quant::UniformQuantizedPerAxisType>(outQ))
      return failure();

    // Forward: quant operand, f32 result.
    if (inQ && isF32Tensor(out.getType())) {
      SmallVector<func::ReturnOp> returnUsers;
      for (Operation *u : out.getUsers()) {
        if (auto r = dyn_cast<func::ReturnOp>(u)) {
          returnUsers.push_back(r);
          continue;
        }
        // Any other type-bound consumer: skip the whole rewrite.
        if (isTypeBoundConsumer(u))
          return failure();
      }
      auto uQ = dyn_cast<quant::UniformQuantizedType>(inQ);
      if (!returnUsers.empty() && !uQ)
        return failure();
      Type newOutTy = cast<TensorType>(out.getType()).clone(inQ);
      rewriter.modifyOpInPlace(op, [&]() { out.setType(newOutTy); });
      for (func::ReturnOp r : returnUsers) {
        rewriter.setInsertionPoint(r);
        Value bridge = insertQuantToF32Bridge(rewriter, op->getLoc(), out, uQ);
        rewriter.modifyOpInPlace(r, [&]() {
          for (OpOperand &opnd : r->getOpOperands())
            if (opnd.get() == out)
              opnd.set(bridge);
        });
      }
      return success();
    }

    // Backward: f32 operand, quant result.
    if (outQ && isF32Tensor(in.getType())) {
      auto uQ = dyn_cast<quant::UniformQuantizedType>(outQ);
      // Graph input bridge: operand is a function block argument.
      if (isa<BlockArgument>(in)) {
        if (!uQ)
          return failure();
        rewriter.setInsertionPoint(op);
        Value bridge = insertF32ToQuantBridge(rewriter, op->getLoc(), in, uQ);
        rewriter.modifyOpInPlace(op, [&]() { op->setOperand(0, bridge); });
        return success();
      }
      Operation *producer = in.getDefiningOp();
      if (!producer || isTypeBoundProducer(producer))
        return failure();
      // Skip only if a sibling user is non-return type-bound. func.return
      // siblings are bridged to keep the graph output at f32.
      SmallVector<func::ReturnOp> returnSiblings;
      for (Operation *u : in.getUsers()) {
        if (u == op)
          continue;
        if (auto r = dyn_cast<func::ReturnOp>(u)) {
          returnSiblings.push_back(r);
          continue;
        }
        if (isTypeBoundConsumer(u))
          return failure();
      }
      if (!returnSiblings.empty() && !uQ)
        return failure();
      Type newInTy = cast<TensorType>(in.getType()).clone(outQ);
      rewriter.modifyOpInPlace(producer, [&]() { in.setType(newInTy); });
      for (func::ReturnOp r : returnSiblings) {
        rewriter.setInsertionPoint(r);
        Value bridge = insertQuantToF32Bridge(rewriter, op->getLoc(), in, uQ);
        rewriter.modifyOpInPlace(r, [&]() {
          for (OpOperand &opnd : r->getOpOperands())
            if (opnd.get() == in)
              opnd.set(bridge);
        });
      }
      return success();
    }

    return failure();
  }
};

struct PropagateQuantTypeThroughDataFlowPass
    : public PassWrapper<PropagateQuantTypeThroughDataFlowPass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "propagate-quant-type-through-dataflow";
  }
  StringRef getDescription() const override {
    return "Unify f32 <-> !quant.uniform across pure data-flow ops "
           "(Reshape/Transpose/Squeeze/Unsqueeze/Flatten/Identity/"
           "DepthToSpace/SpaceToDepth/ReverseSequence). Function inputs and "
           "outputs are bridged with QuantizeLinear+scast or scast+"
           "DequantizeLinear so the ABI stays f32.";
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

} // namespace

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createPropagateQuantTypeThroughDataFlowPass() {
  return std::make_unique<PropagateQuantTypeThroughDataFlowPass>();
}

} // namespace onnx_mlir
