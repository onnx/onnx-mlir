/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--- ZHighConstructSDPA3D.cpp - Fuse Attention Pattern ----------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This pass fuses MatMul(Softmax(MatMul(Q, KT) + Mask), V) into a single
// zhigh.SDPA3D operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace {

struct ConstructSDPA3DPattern : public OpRewritePattern<ZHighMatMulOp> {
  using OpRewritePattern<ZHighMatMulOp>::OpRewritePattern;

  // Insert ZHighUnstick -> ONNXTranspose(perm=[0,2,1]) -> ZHighStick(3DS)
  // to swap the last two dimensions of a 3DS tensor.
  Value transpose3DS(
      Value input, Location loc, PatternRewriter &rewriter) const {
    Value unsticked = ZHighUnstickOp::create(rewriter, loc, input);
    auto inType = mlir::cast<RankedTensorType>(unsticked.getType());
    ArrayRef<int64_t> inShape = inType.getShape();
    SmallVector<int64_t, 3> transposedShape(
        {inShape[0], inShape[2], inShape[1]});
    RankedTensorType transposedType =
        RankedTensorType::get(transposedShape, inType.getElementType());
    ArrayAttr perm = rewriter.getI64ArrayAttr({0, 2, 1});
    Value transposed =
        ONNXTransposeOp::create(rewriter, loc, transposedType, unsticked, perm);
    StringAttr layout = rewriter.getStringAttr("3DS");
    return ZHighStickOp::create(
        rewriter, loc, transposed, layout, /*saturation=*/IntegerAttr());
  }

  LogicalResult matchAndRewrite(
      ZHighMatMulOp trailingMatmul, PatternRewriter &rewriter) const override {
    // Trailing MatMul: no transpose, no bias.
    if (trailingMatmul.getTransposeA() != 0 ||
        trailingMatmul.getTransposeB() != 0)
      return failure();
    if (!mlir::isa<NoneType>(trailingMatmul.getB().getType()))
      return failure();

    // V (RHS of trailing MatMul) must be 3DS.
    Value V = trailingMatmul.getY();
    if (getZTensorLayout(V.getType()) != ZTensorEncodingAttr::DataLayout::_3DS)
      return failure();

    // LHS must come from a single-use Softmax (ACT_NONE).
    Value softmaxResult = trailingMatmul.getX();
    auto softmaxOp = softmaxResult.getDefiningOp<ZHighSoftmaxOp>();
    if (!softmaxOp || !softmaxOp.getResult().hasOneUse())
      return failure();
    if (softmaxOp.getActFunc() != "ACT_NONE")
      return failure();

    // Softmax input must come from a single-use Add.
    Value softmaxInput = softmaxOp.getX();
    auto addOp = softmaxInput.getDefiningOp<ZHighAddOp>();
    if (!addOp || !addOp.getResult().hasOneUse())
      return failure();

    // One Add operand must come from a single-use MatMul (inner);
    // the other is the mask.
    Value addLHS = addOp.getX();
    Value addRHS = addOp.getY();
    auto innerMatmul = addLHS.getDefiningOp<ZHighMatMulOp>();
    Value mask = addRHS;
    if (!innerMatmul) {
      innerMatmul = addRHS.getDefiningOp<ZHighMatMulOp>();
      mask = addLHS;
    }
    if (!innerMatmul || !innerMatmul.getResult().hasOneUse())
      return failure();

    // Inner MatMul: no bias.
    if (!mlir::isa<NoneType>(innerMatmul.getB().getType()))
      return failure();

    // All operands must be 3DS.
    Value Q = innerMatmul.getX();
    Value KT = innerMatmul.getY();
    if (getZTensorLayout(Q.getType()) != ZTensorEncodingAttr::DataLayout::_3DS)
      return failure();
    if (getZTensorLayout(KT.getType()) != ZTensorEncodingAttr::DataLayout::_3DS)
      return failure();
    if (getZTensorLayout(mask.getType()) !=
        ZTensorEncodingAttr::DataLayout::_3DS)
      return failure();

    // Handle transposes on the inner MatMul operands.
    // SDPA3D expects Q=[s,m,n] and KT=[s,n,p].
    Location loc = trailingMatmul.getLoc();
    if (innerMatmul.getTransposeA() != 0)
      Q = transpose3DS(Q, loc, rewriter);
    if (innerMatmul.getTransposeB() != 0)
      KT = transpose3DS(KT, loc, rewriter);

    // Replace the trailing MatMul with the fused op.
    rewriter.replaceOpWithNewOp<ZHighSDPA3DOp>(trailingMatmul, Q, KT, mask, V);
    return success();
  }
};

struct ZHighConstructSDPA3DPass : public PassWrapper<ZHighConstructSDPA3DPass,
                                      OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighConstructSDPA3DPass)

  StringRef getArgument() const override { return "zhigh-construct-sdpa3d"; }

  StringRef getDescription() const override {
    return "Fuse MatMul(Softmax(MatMul(Q,KT)+Mask),V) into "
           "zhigh.SDPA3D";
  }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<ConstructSDPA3DPattern>(context);

    if (failed(applyPatternsGreedily(function, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace onnx_mlir {
namespace zhigh {

std::unique_ptr<mlir::Pass> createZHighConstructSDPA3DPass() {
  return std::make_unique<ZHighConstructSDPA3DPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
