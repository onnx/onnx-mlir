// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Transfer batch dimension on quantized onnx.XCOMPILERFusedEltwise into
// leading reshape + eltwise + trailing reshape. Mirrors xcompiler's
// TransferBatchEltwiseToReshapeEltwisePass for the MLIR representation.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ResultNamesUpdater.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;

namespace {

static bool dim0NeedsTransfer(RankedTensorType rt) {
  if (ShapedType::isDynamic(rt.getShape()[0]))
    return false;
  return rt.getShape()[0] != 1;
}

static std::optional<int64_t> staticTensorVolume(ArrayRef<int64_t> shape) {
  int64_t v = 1;
  for (int64_t d : shape) {
    if (ShapedType::isDynamic(d))
      return std::nullopt;
    v *= d;
  }
  return v;
}

struct TransferBatchXCompilerFusedEltwisePattern
    : public OpRewritePattern<XCOMPILERFusedEltwiseOp> {
  using OpRewritePattern<XCOMPILERFusedEltwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XCOMPILERFusedEltwiseOp fusedOp,
      PatternRewriter &rewriter) const override {
    Location loc = fusedOp.getLoc();

    auto outTy = dyn_cast<RankedTensorType>(fusedOp.getResult().getType());
    if (!outTy || outTy.getRank() != 4)
      return failure();
    if (!isa<quant::QuantizedType>(outTy.getElementType()))
      return rewriter.notifyMatchFailure(fusedOp, "result not ranked 4D quant");

    Value aVal = fusedOp.getA();
    Value bVal = fusedOp.getB();
    bool hasB = bVal && !isa<NoneType>(bVal.getType());

    auto aTy = dyn_cast<RankedTensorType>(aVal.getType());
    if (!aTy || aTy.getRank() != 4 ||
        !isa<quant::QuantizedType>(aTy.getElementType()))
      return failure();

    RankedTensorType bTy;
    if (hasB) {
      bTy = dyn_cast<RankedTensorType>(bVal.getType());
      if (!bTy || bTy.getRank() != 4 ||
          !isa<quant::QuantizedType>(bTy.getElementType()))
        return failure();
    }

    if (!dim0NeedsTransfer(aTy) && (!hasB || !dim0NeedsTransfer(bTy)))
      return rewriter.notifyMatchFailure(
          fusedOp, "leading batch dim is 1 on all operands");

    if (ShapedType::isDynamic(aTy.getShape()[2]) ||
        ShapedType::isDynamic(aTy.getShape()[3]))
      return rewriter.notifyMatchFailure(fusedOp, "A has dynamic H/W");
    int64_t fuseProd = aTy.getShape()[2] * aTy.getShape()[3];
    if (hasB) {
      if (ShapedType::isDynamic(bTy.getShape()[2]) ||
          ShapedType::isDynamic(bTy.getShape()[3]))
        return rewriter.notifyMatchFailure(fusedOp, "B has dynamic H/W");
      if (bTy.getShape()[2] * bTy.getShape()[3] != fuseProd)
        return rewriter.notifyMatchFailure(fusedOp, "H*W product mismatch");
    }

    SmallVector<int64_t, 4> innerAShape = {
        1, aTy.getShape()[0], aTy.getShape()[1], fuseProd};
    SmallVector<int64_t, 4> innerBShape;
    if (hasB) {
      innerBShape.assign({1, bTy.getShape()[0], bTy.getShape()[1], fuseProd});
    }

    SmallVector<int64_t, 4> innerOutShape;
    if (hasB) {
      if (!OpTrait::util::getBroadcastedShape(
              innerAShape, innerBShape, innerOutShape))
        return rewriter.notifyMatchFailure(
            fusedOp, "inner reshape shapes do not broadcast");
    } else {
      innerOutShape.assign(innerAShape.begin(), innerAShape.end());
    }

    auto innerOutVol = staticTensorVolume(innerOutShape);
    auto outVol = staticTensorVolume(outTy.getShape());
    if (!innerOutVol || !outVol || *innerOutVol != *outVol)
      return rewriter.notifyMatchFailure(
          fusedOp, "inner/outer tensor volume mismatch");

    rewriter.setInsertionPoint(fusedOp);
    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);

    auto reshapeVal = [&](Value v, ArrayRef<int64_t> newShape,
                          RankedTensorType inTy) -> Value {
      auto reshapedElemTy = inTy.getElementType();
      auto reshapedTy = RankedTensorType::get(newShape, reshapedElemTy);
      Value shapeConst = onnxBuilder.constantInt64(newShape);
      return rewriter.create<ONNXReshapeOp>(loc, reshapedTy, v, shapeConst)
          .getResult();
    };

    Value newA = reshapeVal(aVal, innerAShape, aTy);
    Value newB;
    if (hasB)
      newB = reshapeVal(bVal, innerBShape, bTy);
    else
      newB = rewriter.create<ONNXNoneOp>(loc).getResult();

    Type innerResultElemTy = outTy.getElementType();
    auto innerResultTy =
        RankedTensorType::get(innerOutShape, innerResultElemTy);

    Operation *orig = fusedOp.getOperation();
    OperationState state(loc, orig->getName());
    state.addTypes(innerResultTy);
    state.addOperands(ValueRange{newA, newB});
    for (NamedAttribute na : orig->getAttrs())
      state.addAttribute(na.getName(), na.getValue());
    Operation *innerFusedPtr = rewriter.create(state);
    auto innerFused = cast<XCOMPILERFusedEltwiseOp>(innerFusedPtr);

    rewriter.setInsertionPointAfter(innerFused);
    SmallVector<int64_t, 4> postShape(
        outTy.getShape().begin(), outTy.getShape().end());
    Value postShapeConst = onnxBuilder.constantInt64(postShape);
    Value postReshape = rewriter
                            .create<ONNXReshapeOp>(loc, outTy,
                                innerFused.getResult(), postShapeConst)
                            .getResult();

    rewriter.replaceOp(fusedOp, postReshape);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

struct TransferBatchXCompilerFusedEltwisePass
    : public PassWrapper<TransferBatchXCompilerFusedEltwisePass,
          OperationPass<func::FuncOp>> {
  StringRef getArgument() const override {
    return "transfer-batch-xcompiler-fused-eltwise";
  }
  StringRef getDescription() const override {
    return "Fold batch on quantized onnx.XCOMPILERFusedEltwise into "
           "reshape + eltwise + reshape (XMC)";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<TransferBatchXCompilerFusedEltwisePattern>(context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    ResultNamesUpdater rnUpdater;
    config.listener = &rnUpdater;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createTransferBatchXCompilerFusedEltwisePass() {
  return std::make_unique<TransferBatchXCompilerFusedEltwisePass>();
}

} // namespace onnx_mlir
