/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ DepthToSpace.cpp - Lowering DepthToSpace Op -------------===//
//
// Copyright 2021-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX DepthToSpace Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "depth_to_space_onnx_to_krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXDepthToSpaceOpLowering : public ConversionPattern {
  ONNXDepthToSpaceOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXDepthToSpaceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto spaceToDepthOp = llvm::cast<ONNXDepthToSpaceOp>(op);
    // assert(spaceToDepthOp && "Expecting op to have type ONNXDepthToSpaceOp");
    ONNXDepthToSpaceOpAdaptor operandAdaptor(operands);
    Value input = operandAdaptor.input();
    Location loc = op->getLoc();
    MultiDialectBuilder<IndexExprBuilderForKrnl, OnnxToKrnlBuilder> create(
        rewriter, loc);

    // Get shape.
    NewONNXDepthToSpaceOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    int64_t bs = spaceToDepthOp.blocksize();
    StringRef mode = spaceToDepthOp.mode();
    assert(create.krnlIE.getTypeRank(input) == 4 &&
           "Input tensor should have rank equal to 4");

    // Compute the new dimensions.
    // MemRefBoundsIndexCapture bounds(input);

    DimIndexExpr B(create.krnlIE.getShapeAsDim(input, 0));
    DimIndexExpr C(create.krnlIE.getShapeAsDim(input, 1));
    DimIndexExpr H(create.krnlIE.getShapeAsDim(input, 2));
    DimIndexExpr W(create.krnlIE.getShapeAsDim(input, 3));
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

    // OnnxToKrnlBuilder create(rewriter, loc);

    // Reshape input tensor to shape:
    //   [B, bs, bs, C/(bs*bs), H, W] when mode=DCR
    //   [B, C/(bs*bs), bs, bs, H, W] when mode=CRD
    Value reshapeRes1 = create.krnlOnnx.reshape(input, outputDims1);
    LLVM_DEBUG(llvm::dbgs() << "reshapeRes1: " << reshapeRes1 << "\n");

    // Transpose the reshape result into shape [B, C/(bs*bs), H, bs, W, bs].
    SmallVector<DimIndexExpr> outputDims2({B, newC, H, bsLit, W, bsLit});
    Value transposeRes =
        create.krnlOnnx.transpose(reshapeRes1, perm, outputDims2);
    LLVM_DEBUG(llvm::dbgs() << "transposeRes: " << transposeRes << "\n");

    // Reshape the transpose result into shape [B, C/(bs*bs), H*bs, W*bs].
    SmallVector<DimIndexExpr> outputDims3({B, newC, newH, newW});
    Value reshapeRes2 = create.krnlOnnx.reshape(transposeRes, outputDims3);
    LLVM_DEBUG(llvm::dbgs() << "reshapeRes2: " << reshapeRes2 << "\n");

    rewriter.replaceOp(op, reshapeRes2);

    return success();
  }
};

void populateLoweringONNXDepthToSpaceOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDepthToSpaceOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
