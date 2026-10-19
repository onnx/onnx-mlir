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

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace {

// Fuse the standard scaled dot-product attention (SDPA) subgraph:
//
//   trailing MatMul
//       |      \
//    Softmax    V
//       |
//      Add
//      / \
//  inner   Mask
//  MatMul
//   / \
//  Q   KT (or K when transposeB=1)
//
// into a single zhigh.SDPA3D op:
//   Output = SDPA3D(Q, KT, Mask, V, transposeKT)
//
// Constraints:
//   - All operands (Q, KT/K, Mask, V) must be 3DS layout.
//   - Both MatMuls must have no bias.
//   - Trailing MatMul must have transposeA=0 and transposeB=0.
//   - Inner MatMul must have transposeA=0; transposeB is propagated as
//     the transposeKT attribute so the runtime can use NNPA hardware
//     transpose (zero-cost, no data copy).
//   - Softmax must use ACT_NONE.
//   - All intermediate results must be single-use.
struct ConstructSDPA3DPattern : public OpRewritePattern<ZHighMatMulOp> {
  using OpRewritePattern<ZHighMatMulOp>::OpRewritePattern;

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

    // Inner MatMul: no transposeA, no bias.
    if (innerMatmul.getTransposeA() != 0)
      return failure();
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

    // Pass transposeKT through to SDPA3D — runtime handles the transpose
    // inside zdnn_matmul_transpose_op (zero-cost NNPA hardware transpose).
    IntegerAttr transposeKTAttr = rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, /*isSigned=*/true),
        innerMatmul.getTransposeB() != 0 ? 1 : 0);

    // Replace the trailing MatMul with the fused op.
    rewriter.replaceOpWithNewOp<ZHighSDPA3DOp>(
        trailingMatmul, Q, KT, mask, V, transposeKTAttr);
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
