/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ DepthToSpace.cpp - Lowering DepthToSpace Op -------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX DepthToSpace Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "llvm/IR/Constants.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using llvm::dbgs;

#define DEBUG_TYPE "depth_to_space_onnx_to_krnl"

struct ONNXDepthToSpaceOpLowering : public ConversionPattern {
  ONNXDepthToSpaceOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXDepthToSpaceOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto spaceToDepthOp = dyn_cast_or_null<ONNXDepthToSpaceOp>(op);
    assert(spaceToDepthOp && "Expecting op to have type ONNXDepthToSpaceOp");

    // Ensure we can compute the operator output shape.
    ONNXDepthToSpaceOpShapeHelper shapeHelper(&spaceToDepthOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    ONNXDepthToSpaceOpAdaptor operandAdaptor(operands);
    LogicalResult shapeComputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapeComputed;
    assert(succeeded(shapeComputed) && "Could not compute output shape");

    Location loc = spaceToDepthOp.getLoc();
    Value input = spaceToDepthOp.input();
    int64_t bs = spaceToDepthOp.blocksize();
    StringRef mode = spaceToDepthOp.mode();

    // Compute the new dimensions.
    MemRefBoundsIndexCapture bounds(input);
    assert(bounds.getRank() == 4 && "Input tensor should have rank equal to 4");

    DimIndexExpr B(bounds.getDim(0));
    DimIndexExpr C(bounds.getDim(1));
    DimIndexExpr H(bounds.getDim(2));
    DimIndexExpr W(bounds.getDim(3));
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
    Value reshapeRes1 = reshape(input, outputDims1, rewriter, loc);

    // Transpose the reshape result into shape [B, C/(bs*bs), H, bs, W, bs].
    SmallVector<DimIndexExpr> outputDims2({B, newC, H, bsLit, W, bsLit});
    Value transposeRes =
        transpose(reshapeRes1, outputDims2, perm, rewriter, loc);

    // Reshape the transpose result into shape [B, C/(bs*bs), H*bs, W*bs].
    SmallVector<DimIndexExpr> outputDims3({B, newC, newH, newW});
    Value reshapeRes2 = reshape(transposeRes, outputDims3, rewriter, loc);

    rewriter.replaceOp(op, reshapeRes2);

    return success();
  }

private:
  // Reshape the 'input' tensor to the shape prodided by 'outputDims'.
  Value reshape(const Value input, const ArrayRef<DimIndexExpr> outputDims,
      ConversionPatternRewriter &rewriter, const Location &loc) const {
    assert(!outputDims.empty() && "Output dimensions should not be empty");

    OnnxBuilder onnxBuilder(rewriter, loc);

    // If the output dimensions are all literals the 'onnx/Reshape' operation
    // can take the new shape via an 'onnx.Constant'.
    if (llvm::all_of(outputDims,
            [](const DimIndexExpr &dim) { return dim.isLiteral(); })) {
      SmallVector<int64_t, 6> shape;
      for (const IndexExpr &dim : outputDims)
        shape.push_back(dim.getLiteral());

      auto constantOp = getONNXConstantOpFromDenseAttr(
          rewriter, loc, rewriter.getI64TensorAttr(shape));
      LLVM_DEBUG(dbgs() << "constantOp: " << constantOp << "\n");

      ShapedType inputType = input.getType().cast<ShapedType>();
      Type elementType = inputType.getElementType();
      Value reshapeRes = onnxBuilder.reshape(
          MemRefType::get(shape, elementType), input, constantOp);
      LLVM_DEBUG(dbgs() << "reshapeRes: " << reshapeRes << "\n");

      return reshapeRes;
    }

    MemRefBuilder memRefBuilder(onnxBuilder);
    KrnlBuilder krnlBuilder(onnxBuilder);

    // When the output dimensions aren't all literals we need to generate code
    // to compute the shape. Allocate a buffer and store the putput dimension
    // into it.
    IndexType indexTy = rewriter.getIndexType();
    int64_t length = outputDims.size();
    memref::AllocOp alloc =
        memRefBuilder.alignedAlloc(MemRefType::get({length}, indexTy), 16);
    LLVM_DEBUG(dbgs() << "alloc: " << alloc << "\n");

    for (int64_t i = 0; i < length; ++i) {
      Value index = emitConstantOp(rewriter, loc, indexTy, i);
      Value data = outputDims[i].getValue();
      krnlBuilder.store(data, alloc, index);
    }

    // Now create the "onnx.Reshape" operation. Because the shape is not a
    // compile time constant it is effectively unknown.
    SmallVector<int64_t> shape(length, -1);
    ShapedType inputType = input.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();
    Value reshapeRes =
        onnxBuilder.reshape(MemRefType::get(shape, elementType), input, alloc);
    LLVM_DEBUG(dbgs() << "reshapeRes: " << reshapeRes << "\n");

    // The 'onnx.Reshape' operation yields a memref with unknown extents, so we
    // need to explicitly cast the result to the know size.
    SmallVector<int64_t, 6> castOutputShape;
    for (const IndexExpr &dim : outputDims)
      castOutputShape.push_back(dim.isLiteral() ? dim.getLiteral() : -1);

    Value castRes = memRefBuilder.cast(
        reshapeRes, MemRefType::get(castOutputShape, elementType));
    LLVM_DEBUG(dbgs() << "castRes: " << castRes << "\n");

    return castRes;
  }

  // Transpose the 'input' tensor given the permutation array.
  Value transpose(const Value input, const ArrayRef<DimIndexExpr> outputDims,
      const ArrayRef<int64_t> perm, ConversionPatternRewriter &rewriter,
      const Location &loc) const {
    assert(!outputDims.empty() && "Output dimensions should not be empty");
    assert(!perm.empty() && perm.size() == outputDims.size() &&
           "Expecitng valid permutation array");

    // Compute the shape of the 'onnx.Transpose' result.
    SmallVector<int64_t, 6> shape;
    for (const IndexExpr &dim : outputDims)
      shape.push_back(dim.isLiteral() ? dim.getLiteral() : -1);

    // Compute the memref type of the "onnx.Transpose" output.
    ShapedType inputType = input.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();
    MemRefType memRefType = MemRefType::get(shape, elementType);

    // Create the "onnx.Transpose" operation.
    OnnxBuilder onnxBuilder(rewriter, loc);
    Value transposeRes = onnxBuilder.transpose(
        memRefType, input, rewriter.getI64ArrayAttr(perm));
    LLVM_DEBUG(dbgs() << "transposeRes: " << transposeRes << "\n");

    return transposeRes;
  }
};

void populateLoweringONNXDepthToSpaceOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXDepthToSpaceOpLowering>(ctx);
}
