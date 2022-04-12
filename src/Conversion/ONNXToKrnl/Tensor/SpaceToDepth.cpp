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
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "space_to_depth_onnx_to_krnl"

using namespace mlir;

namespace onnx_mlir {

struct ONNXSpaceToDepthOpLowering : public ConversionPattern {
  ONNXSpaceToDepthOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXSpaceToDepthOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto spaceToDepthOp = dyn_cast_or_null<ONNXSpaceToDepthOp>(op);
    assert(spaceToDepthOp && "Expecting op to have type ONNXSpaceToDepthOp");

    // Ensure we can compute the operator output shape.
    ONNXSpaceToDepthOpShapeHelper shapeHelper(&spaceToDepthOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    ONNXSpaceToDepthOpAdaptor operandAdaptor(operands);
    LogicalResult shapeComputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapeComputed;
    assert(succeeded(shapeComputed) && "Could not compute output shape");

    Location loc = spaceToDepthOp.getLoc();
    Value input = operandAdaptor.input();
    int64_t bs = spaceToDepthOp.blocksize();

    // Compute the new dimensions.
    MemRefBoundsIndexCapture bounds(input);
    assert(bounds.getRank() == 4 && "Input tensor should have rank equal to 4");

    DimIndexExpr B(bounds.getDim(0));
    DimIndexExpr C(bounds.getDim(1));
    DimIndexExpr H(bounds.getDim(2));
    DimIndexExpr W(bounds.getDim(3));
    DimIndexExpr newC = C * (bs * bs);
    DimIndexExpr newH = H.floorDiv(bs);
    DimIndexExpr newW = W.floorDiv(bs);

    OnnxToKrnlBuilder create(rewriter, loc);

    // Reshape input tensor to shape [B, C, H/bs, bs, W/bs, bs].
    LiteralIndexExpr bsLit(bs);
    SmallVector<DimIndexExpr> outputDims1({B, C, newH, bsLit, newW, bsLit});
    Value reshapeRes1 = create.reshape(input, outputDims1);
    LLVM_DEBUG(llvm::dbgs() << "reshapeRes1: " << reshapeRes1 << "\n");

    // Transpose the reshape result into shape [B, C, bs, bs, H/bs, W/bs].
    SmallVector<DimIndexExpr> outputDims2({B, C, bsLit, bsLit, newH, newW});
    SmallVector<int64_t> perm({0, 1, 3, 5, 2, 4});
    Value transposeRes = create.transpose(reshapeRes1, perm, outputDims2);
    LLVM_DEBUG(llvm::dbgs() << "transposeRes: " << transposeRes << "\n");

    // Reshape the transpose result into shape [B, C*bs*bs, H/bs, W/bs].
    SmallVector<DimIndexExpr> outputDims3({B, newC, newH, newW});
    Value reshapeRes2 = create.reshape(transposeRes, outputDims3);
    LLVM_DEBUG(llvm::dbgs() << "reshapeRes2: " << reshapeRes2 << "\n");

    rewriter.replaceOp(op, reshapeRes2);

    return success();
  }
};

void populateLoweringONNXSpaceToDepthOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSpaceToDepthOpLowering>(typeConverter, ctx);
}
} // namespace onnx_mlir
