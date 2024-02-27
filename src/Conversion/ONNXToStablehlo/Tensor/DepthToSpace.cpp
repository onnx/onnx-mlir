/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ DepthToSpace.cpp - Lowering DepthToSpace Op -------------===//
//
// Copyright 2023-2024
//
// =============================================================================
//
// This file lowers the ONNX DepthToSpace Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct ONNXDepthToSpaceOpLoweringToStablehlo
    : public OpConversionPattern<ONNXDepthToSpaceOp> {
  ONNXDepthToSpaceOpLoweringToStablehlo(MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(ONNXDepthToSpaceOp depthToSpaceOp,
      ONNXDepthToSpaceOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = depthToSpaceOp.getOperation();
    Location loc = ONNXLoc<ONNXDepthToSpaceOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value input = adaptor.getInput();

    MultiDialectBuilder<IndexExprBuilderForStablehlo, OnnxToStablehloBuilder>
        create(rewriter, loc);
    ONNXDepthToSpaceOpShapeHelper shapeHelper(
        op, operands, &create.stableHloIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    int64_t bs = depthToSpaceOp.getBlocksize();
    StringRef mode = depthToSpaceOp.getMode();
    assert(create.stableHloIE.getShapedTypeRank(input) == 4 &&
           "Input tensor should have rank equal to 4");

    // Compute the new dimensions.

    DimIndexExpr B(create.stableHloIE.getShapeAsDim(input, 0));
    DimIndexExpr C(create.stableHloIE.getShapeAsDim(input, 1));
    DimIndexExpr H(create.stableHloIE.getShapeAsDim(input, 2));
    DimIndexExpr W(create.stableHloIE.getShapeAsDim(input, 3));
    DimIndexExpr newC = C.floorDiv(bs * bs);
    DimIndexExpr newH = H * bs;
    DimIndexExpr newW = W * bs;

    // Compute the output dimension of the first reshape operation, and the
    // permutation array for the transpose operation.
    LiteralIndexExpr bsLit(bs);
    SmallVector<DimIndexExpr, 6> outputDims1;
    SmallVector<int64_t, 6> perm;
    if (mode == "DCR") {
      outputDims1 = {B, bsLit, bsLit, newC, H, W};
      perm = {0, 3, 4, 1, 5, 2};
    } else {
      assert(mode == "CRD" && "Unexpected mode");
      outputDims1 = {B, newC, bsLit, bsLit, H, W};
      perm = {0, 1, 4, 2, 5, 3};
    }

    // Reshape input tensor to shape:
    //   [B, bs, bs, C/(bs*bs), H, W] when mode=DCR
    //   [B, C/(bs*bs), bs, bs, H, W] when mode=CRD
    Value reshapeRes1 = create.stablehloOnnx.reshape(input, outputDims1);

    // Transpose the reshape result into shape [B, C/(bs*bs), H, bs, W, bs].
    SmallVector<DimIndexExpr> outputDims2({B, newC, H, bsLit, W, bsLit});
    Value transposeRes =
        create.stablehloOnnx.transpose(reshapeRes1, perm, outputDims2);

    // Reshape the transpose result into shape [B, C/(bs*bs), H*bs, W*bs].
    SmallVector<DimIndexExpr> outputDims3({B, newC, newH, newW});
    Value reshapeRes2 = create.stablehloOnnx.reshape(transposeRes, outputDims3);

    rewriter.replaceOp(op, reshapeRes2);
    return success();
  }
};

} // namespace

void populateLoweringONNXDepthToSpaceOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXDepthToSpaceOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
