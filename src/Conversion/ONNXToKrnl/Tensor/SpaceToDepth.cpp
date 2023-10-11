/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- SpaceToDepth.cpp - Lowering SpaceToDepthOp ----------------===//
//
// Copyright 2021-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SpaceToDepth Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "space_to_depth_onnx_to_krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSpaceToDepthOpLowering
    : public OpConversionPattern<ONNXSpaceToDepthOp> {
  ONNXSpaceToDepthOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXSpaceToDepthOp spaceToDepthOp,
      ONNXSpaceToDepthOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = spaceToDepthOp.getOperation();
    Location loc = ONNXLoc<ONNXSpaceToDepthOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, OnnxToKrnlBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXSpaceToDepthOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Value input = adaptor.getInput();
    int64_t bs = adaptor.getBlocksize();

    // Compute the new dimensions.
    assert(create.krnlIE.getShapedTypeRank(input) == 4 &&
           "Input tensor should have rank equal to 4");

    DimIndexExpr B(create.krnlIE.getShapeAsDim(input, 0));
    DimIndexExpr C(create.krnlIE.getShapeAsDim(input, 1));
    DimIndexExpr H(create.krnlIE.getShapeAsDim(input, 2));
    DimIndexExpr W(create.krnlIE.getShapeAsDim(input, 3));
    DimIndexExpr newC = C * (bs * bs);
    DimIndexExpr newH = H.floorDiv(bs);
    DimIndexExpr newW = W.floorDiv(bs);

    // Reshape input tensor to shape [B, C, H/bs, bs, W/bs, bs].
    LiteralIndexExpr bsLit(bs);
    SmallVector<DimIndexExpr> outputDims1({B, C, newH, bsLit, newW, bsLit});
    Value reshapeRes1 = create.krnlOnnx.reshape(input, outputDims1);
    LLVM_DEBUG(llvm::dbgs() << "reshapeRes1: " << reshapeRes1 << "\n");

    // Transpose the reshape result into shape [B, C, bs, bs, H/bs, W/bs].
    SmallVector<DimIndexExpr> outputDims2({B, C, bsLit, bsLit, newH, newW});
    SmallVector<int64_t> perm({0, 1, 3, 5, 2, 4});
    Value transposeRes =
        create.krnlOnnx.transpose(reshapeRes1, perm, outputDims2);
    LLVM_DEBUG(llvm::dbgs() << "transposeRes: " << transposeRes << "\n");

    // Reshape the transpose result into shape [B, C*bs*bs, H/bs, W/bs].
    SmallVector<DimIndexExpr> outputDims3({B, newC, newH, newW});
    Value reshapeRes2 = create.krnlOnnx.reshape(transposeRes, outputDims3);
    LLVM_DEBUG(llvm::dbgs() << "reshapeRes2: " << reshapeRes2 << "\n");

    rewriter.replaceOp(op, reshapeRes2);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXSpaceToDepthOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSpaceToDepthOpLowering>(typeConverter, ctx);
}
} // namespace onnx_mlir
