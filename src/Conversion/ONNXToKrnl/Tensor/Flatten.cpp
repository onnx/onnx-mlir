/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Flatten.cpp - Lowering Flatten Op -------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Flatten Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Helper function to insert alloc and dealloc ops for memref of dynamic shape.
//
// Should namespace or static be used here?
Value insertAllocForFlatten(MemRefType memRefType, Location loc,
    ConversionPatternRewriter &rewriter, Value input, int64_t axisValue) {
  MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);
  memref::AllocOp alloc;
  auto inputShape = mlir::cast<MemRefType>(input.getType()).getShape();
  int64_t inputRank = inputShape.size();

  SmallVector<Value, 2> allocOperands;
  // Compute size for the first dimension when not constant
  if (memRefType.isDynamicDim(0)) {
    Value dimVal = create.math.constantIndex(1);
    for (int64_t i = 0; i < axisValue; i++)
      dimVal = create.math.mul(dimVal, create.mem.dim(input, i));
    allocOperands.emplace_back(dimVal);
  }

  // Compute size for the second dimension when not constant
  if (memRefType.isDynamicDim(1)) {
    Value dimVal = create.math.constantIndex(1);
    for (int64_t i = axisValue; i < inputRank; i++)
      dimVal = create.math.mul(dimVal, create.mem.dim(input, i));
    allocOperands.emplace_back(dimVal);
  }

  return create.mem.alignedAlloc(memRefType, allocOperands);
}

struct ONNXFlattenOpLowering : public OpConversionPattern<ONNXFlattenOp> {
  ONNXFlattenOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXFlattenOp flattenOp,
      ONNXFlattenOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    // Gather info.
    Operation *op = flattenOp.getOperation();
    Location loc = ONNXLoc<ONNXFlattenOp>(op);

    Value input = adaptor.getInput();
    auto inputTy = mlir::cast<MemRefType>(input.getType());
    auto inputShape = inputTy.getShape();
    size_t inputRank = inputShape.size();
    int64_t axisValue = flattenOp.getAxis();
    if (axisValue < 0)
      axisValue = inputRank + axisValue;
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder> create(rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);

    // Insert alloc and dealloc
    Value alloc = (hasAllConstantDimensions(outputMemRefType))
                      ? create.mem.alignedAlloc(outputMemRefType)
                      : insertAllocForFlatten(
                            outputMemRefType, loc, rewriter, input, axisValue);

    // Define loops and iteration trip counts (equivalent to size of input)
    ValueRange indices;
    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, inputRank);
    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack pack(rewriter, originalLoops);
    for (size_t i = 0; i < inputRank; ++i)
      addDimensionToPack(rewriter, loc, pack, input, i);

    // Create the loops
    KrnlIterateOp iterateOp = create.krnl.iterate(pack);
    Block &iterationBlock = iterateOp.getBodyRegion().front();

    // Now perform the insertions into the body of the just generated loops.
    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Generate the load of input
    SmallVector<Value, 4> inputMemRefVal(iterationBlock.getArguments().begin(),
        iterationBlock.getArguments().end());
    Value inputVal = create.krnl.load(input, inputMemRefVal);

    // Generate the store for output
    // Define affine map for first dim of output
    AffineExpr firstIndexAE = rewriter.getAffineConstantExpr(0);
    AffineExpr firstAccumulatedDimSizeAE = rewriter.getAffineConstantExpr(1);
    for (int64_t i = axisValue - 1; i >= 0; i--) {
      AffineExpr dimIndexAE = rewriter.getAffineDimExpr(i);
      firstIndexAE = firstIndexAE + dimIndexAE * firstAccumulatedDimSizeAE;
      AffineExpr dimSizeAE = rewriter.getAffineSymbolExpr(i);
      firstAccumulatedDimSizeAE = dimSizeAE * firstAccumulatedDimSizeAE;
    }
    AffineMap firstDimMap = AffineMap::get(axisValue, axisValue, firstIndexAE);

    // Create the parameter lists for the affine map
    MemRefBuilder createMemRef(rewriter, loc);
    SmallVector<Value, 4> firstMapArgList;
    for (int64_t i = 0; i < axisValue; i++)
      firstMapArgList.emplace_back(iterationBlock.getArguments()[i]);

    for (int64_t i = 0; i < axisValue; i++)
      firstMapArgList.emplace_back(createMemRef.dim(input, i));

    auto firstDimVal = rewriter.create<affine::AffineApplyOp>(
        loc, firstDimMap, firstMapArgList);

    // Generate index for second dim of output
    AffineExpr secondIndexAE = rewriter.getAffineConstantExpr(0);
    AffineExpr secondAccumulatedDimSizeAE = rewriter.getAffineConstantExpr(1);
    // Can not use auto for i here because i may be negative
    for (int64_t i = inputRank - 1; i >= axisValue; i--) {
      int64_t idx = i - axisValue;
      AffineExpr dimIndexAE = rewriter.getAffineDimExpr(idx);
      secondIndexAE = secondIndexAE + dimIndexAE * secondAccumulatedDimSizeAE;
      AffineExpr dimSizeAE = rewriter.getAffineSymbolExpr(idx);
      secondAccumulatedDimSizeAE = dimSizeAE * secondAccumulatedDimSizeAE;
    }
    AffineMap secondDimMap = AffineMap::get(
        inputRank - axisValue, inputRank - axisValue, secondIndexAE);

    // Create the parameter lists for the affine map
    SmallVector<Value, 4> secondMapArgList;
    for (size_t i = axisValue; i < inputRank; i++)
      secondMapArgList.emplace_back(iterationBlock.getArguments()[i]);
    for (size_t i = axisValue; i < inputRank; i++)
      secondMapArgList.emplace_back(createMemRef.dim(input, i));

    auto secondDimVal = rewriter.create<affine::AffineApplyOp>(
        loc, secondDimMap, secondMapArgList);

    // Create the store
    SmallVector<Value, 2> outputMemRefVal = {firstDimVal, secondDimVal};
    if (hasAllConstantDimensions(outputMemRefType))
      create.krnl.store(inputVal, alloc, outputMemRefVal);
    else
      create.krnl.store(inputVal, alloc, outputMemRefVal);

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXFlattenOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
